/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/*
 * ============================================================================
 * 辅助函数
 * ============================================================================
 */

/* 从 VRAM 读取 32 位指令 */
static uint32_t vram_readl(GPGPUState *s, uint32_t addr)
{
    if (addr + 4 > s->vram_size) {
        return 0;
    }
    return *(uint32_t *)(s->vram_ptr + addr);
}

/* 写 VRAM 32 位 */
static void vram_writel(GPGPUState *s, uint32_t addr, uint32_t val)
{
    if (addr + 4 > s->vram_size) {
        return;
    }
    *(uint32_t *)(s->vram_ptr + addr) = val;
}

/* 计算 Block 中的总线程数 */
static uint32_t calc_block_threads(GPGPUState *s)
{
    return s->kernel.block_dim[0] * s->kernel.block_dim[1] * s->kernel.block_dim[2];
}

/* 从 Block ID 计算线性索引 */
static uint32_t block_id_to_linear(GPGPUState *s, uint32_t block_x, uint32_t block_y, uint32_t block_z)
{
    return block_z * s->kernel.grid_dim[0] * s->kernel.grid_dim[1] +
           block_y * s->kernel.grid_dim[0] +
           block_x;
}

/*
 * ============================================================================
 * RV32I 指令解码
 * ============================================================================
 */

/* 指令字段提取宏 */
#define OPCODE(inst)     ((inst) & 0x7F)
#define RD(inst)          (((inst) >> 7) & 0x1F)
#define RS1(inst)         (((inst) >> 15) & 0x1F)
#define RS2(inst)         (((inst) >> 20) & 0x1F)
#define FUNCT3(inst)      (((inst) >> 12) & 0x07)
#define FUNCT7(inst)      (((inst) >> 25) & 0x7F)
#define IMM_I(inst)       (((int32_t)(inst)) >> 20)
#define IMM_S(inst)       ((((int32_t)(inst) >> 20) & 0xFE0) | (((inst) >> 7) & 0x1F))
#define IMM_SB(inst)      ((((int32_t)(inst) >> 19) & 0x1000) | (((inst) >> 20) & 0x800) | (((inst) >> 20) & 0x1E) | (((inst) >> 7) & 0x1E0))
#define IMM_U(inst)       ((int32_t)(inst) & 0xFFFFF000)
#define IMM_UJ(inst)      ((((int32_t)(inst) >> 20) & 0x80000) | (((inst) >> 20) & 0x7FE) | (((inst) >> 21) & 0x1000) | (((inst) >> 21) & 0xFF000))

/* R-type 操作码 */
#define OPCODE_OP      0x33
#define OPCODE_OP_32   0x3B

/* I-type 操作码 */
#define OPCODE_JALR    0x67
#define OPCODE_LOAD    0x03
#define OPCODE_OP_IMM  0x13
#define OPCODE_OP_IMM_32 0x1B

/* S-type 操作码 */
#define OPCODE_STORE   0x23

/* B-type 操作码 */
#define OPCODE_BRANCH  0x63

/* U-type 操作码 */
#define OPCODE_LUI     0x37
#define OPCODE_AUIPC   0x17

/* J-type 操作码 */
#define OPCODE_JAL     0x6F

/* SYSTEM 操作码 */
#define OPCODE_SYSTEM  0x73

/* funct3 值 */
#define F3_ADDI   0x0
#define F3_SLTI   0x2
#define F3_SLTIU  0x3
#define F3_XORI   0x4
#define F3_ORI    0x6
#define F3_ANDI   0x7
#define F3_SLLI   0x1
#define F3_SRLI   0x5
#define F3_SRAI   0x5
#define F3_ADD    0x0
#define F3_SUB    0x0
#define F3_SLL    0x1
#define F3_SLT    0x2
#define F3_SLTU   0x3
#define F3_XOR    0x4
#define F3_SRL    0x5
#define F3_SRA    0x5
#define F3_OR     0x6
#define F3_AND    0x7
#define F3_BEQ    0x0
#define F3_BNE    0x1
#define F3_BLT    0x4
#define F3_BGE    0x5
#define F3_BLTU   0x6
#define F3_BGEU   0x7

/*
 * ============================================================================
 * RV32I 指令执行
 * ============================================================================
 */

typedef struct {
    bool stop;      /* 是否停止执行 */
    bool halted;    /* 是否遇到 ebreak */
} ExecResult;

static void exec_lane(GPGPUState *s, GPGPULane *lane, ExecResult *result)
{
    uint32_t inst = vram_readl(s, lane->pc);
    uint32_t opcode = OPCODE(inst);
    uint32_t rd = RD(inst);
    uint32_t rs1 = RS1(inst);
    uint32_t rs2 = RS2(inst);
    uint32_t funct3 = FUNCT3(inst);
    uint32_t funct7 = FUNCT7(inst);
    
    if (!lane->active) {
        return;
    }

    lane->pc += 4;

    switch (opcode) {
    case OPCODE_LUI:
        /* LUI: rd = imm << 12 */
        if (rd != 0) {
            lane->gpr[rd] = IMM_U(inst);
        }
        break;

    case OPCODE_AUIPC:
        /* AUIPC: rd = pc + (imm << 12) */
        if (rd != 0) {
            lane->gpr[rd] = lane->pc - 4 + IMM_U(inst);
        }
        break;

    case OPCODE_JAL:
        /* JAL: rd = pc + 4; pc = pc + imm */
        {
            int32_t offset = (int32_t)IMM_UJ(inst);
            if (rd != 0) {
                lane->gpr[rd] = lane->pc;
            }
            lane->pc += offset - 4;
        }
        break;

    case OPCODE_JALR:
        /* JALR: rd = pc + 4; pc = rs1 + imm */
        {
            int32_t offset = IMM_I(inst);
            uint32_t temp = lane->pc;
            lane->pc = (lane->gpr[rs1] + offset) & ~1;
            if (rd != 0) {
                lane->gpr[rd] = temp;
            }
        }
        break;

    case OPCODE_BRANCH:
        /* 条件分支 */
        {
            int32_t offset = (int32_t)IMM_SB(inst);
            uint32_t pc_target = lane->pc + offset - 4;
            bool taken = false;
            
            switch (funct3) {
            case F3_BEQ:  taken = (lane->gpr[rs1] == lane->gpr[rs2]); break;
            case F3_BNE:  taken = (lane->gpr[rs1] != lane->gpr[rs2]); break;
            case F3_BLT:  taken = ((int32_t)lane->gpr[rs1] < (int32_t)lane->gpr[rs2]); break;
            case F3_BGE:  taken = ((int32_t)lane->gpr[rs1] >= (int32_t)lane->gpr[rs2]); break;
            case F3_BLTU: taken = (lane->gpr[rs1] < lane->gpr[rs2]); break;
            case F3_BGEU: taken = (lane->gpr[rs1] >= lane->gpr[rs2]); break;
            }
            
            if (taken) {
                lane->pc = pc_target;
            }
        }
        break;

    case OPCODE_LOAD:
        /* 加载指令 */
        {
            int32_t offset = IMM_I(inst);
            uint32_t addr = lane->gpr[rs1] + offset;
            
            if (funct3 == 0x2) { /* LW */
                if (rd != 0) {
                    lane->gpr[rd] = vram_readl(s, addr);
                }
            } else if (funct3 == 0x0) { /* LB */
                if (rd != 0) {
                    int8_t val = (int8_t)s->vram_ptr[addr];
                    lane->gpr[rd] = (uint32_t)val;
                }
            } else if (funct3 == 0x1) { /* LH */
                if (rd != 0) {
                    int16_t val = *(int16_t *)(s->vram_ptr + addr);
                    lane->gpr[rd] = (uint32_t)val;
                }
            } else if (funct3 == 0x4) { /* LBU */
                if (rd != 0) {
                    lane->gpr[rd] = (uint8_t)s->vram_ptr[addr];
                }
            } else if (funct3 == 0x5) { /* LHU */
                if (rd != 0) {
                    lane->gpr[rd] = *(uint16_t *)(s->vram_ptr + addr);
                }
            }
        }
        break;

    case OPCODE_STORE:
        /* 存储指令 */
        {
            int32_t offset = IMM_S(inst);
            uint32_t addr = lane->gpr[rs1] + offset;
            
            if (funct3 == 0x2) { /* SW */
                vram_writel(s, addr, lane->gpr[rs2]);
            } else if (funct3 == 0x0) { /* SB */
                s->vram_ptr[addr] = (uint8_t)lane->gpr[rs2];
            } else if (funct3 == 0x1) { /* SH */
                *(uint16_t *)(s->vram_ptr + addr) = (uint16_t)lane->gpr[rs2];
            }
        }
        break;

    case OPCODE_OP_IMM:
        /* 立即数运算 */
        {
            int32_t imm = IMM_I(inst);
            switch (funct3) {
            case F3_ADDI:  if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] + imm; break;
            case F3_SLTI:  if (rd != 0) lane->gpr[rd] = ((int32_t)lane->gpr[rs1] < imm) ? 1 : 0; break;
            case F3_SLTIU: if (rd != 0) lane->gpr[rd] = (lane->gpr[rs1] < (uint32_t)imm) ? 1 : 0; break;
            case F3_XORI:  if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] ^ imm; break;
            case F3_ORI:   if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] | imm; break;
            case F3_ANDI:  if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] & imm; break;
            case F3_SLLI:  if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] << (imm & 0x1F); break;
            case F3_SRLI:
                if (rd != 0) {
                    if (funct7 == 0) {
                        lane->gpr[rd] = lane->gpr[rs1] >> (imm & 0x1F);
                    } else if (funct7 == 0x20) {
                        /* SRAI */
                        if (rd != 0) {
                            lane->gpr[rd] = (int32_t)lane->gpr[rs1] >> (imm & 0x1F);
                        }
                    }
                }
                break;
            }
        }
        break;

    case OPCODE_OP:
        /* R-type 运算 */
        {
            switch (funct3) {
            case F3_ADD:
                if (funct7 == 0) {
                    if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] + lane->gpr[rs2];
                } else if (funct7 == 0x20) {
                    if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] - lane->gpr[rs2];
                }
                break;
            case F3_SLL:
                if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] << (lane->gpr[rs2] & 0x1F);
                break;
            case F3_SLT:
                if (rd != 0) lane->gpr[rd] = ((int32_t)lane->gpr[rs1] < (int32_t)lane->gpr[rs2]) ? 1 : 0;
                break;
            case F3_SLTU:
                if (rd != 0) lane->gpr[rd] = (lane->gpr[rs1] < lane->gpr[rs2]) ? 1 : 0;
                break;
            case F3_XOR:
                if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] ^ lane->gpr[rs2];
                break;
            case F3_SRL:
                if (funct7 == 0) {
                    if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] >> (lane->gpr[rs2] & 0x1F);
                } else if (funct7 == 0x20) {
                    if (rd != 0) lane->gpr[rd] = (int32_t)lane->gpr[rs1] >> (lane->gpr[rs2] & 0x1F);
                }
                break;
            case F3_OR:
                if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] | lane->gpr[rs2];
                break;
            case F3_AND:
                if (rd != 0) lane->gpr[rd] = lane->gpr[rs1] & lane->gpr[rs2];
                break;
            }
        }
        break;

    case OPCODE_SYSTEM:
        /* SYSTEM 指令 (ECALL/EBREAK/CSRRW/CSRS/CSSRC等) */
        {
            if (inst == 0x00100073) { /* EBREAK */
                result->stop = true;
                result->halted = true;
            } else if (funct3 == 0x1) { /* CSRRW */
                uint32_t csr_addr = (inst >> 20) & 0xFFF;
                uint32_t temp = 0;
                
                if (csr_addr == CSR_MHARTID) {
                    temp = lane->mhartid;
                    if (rd != 0) {
                        lane->gpr[rd] = temp;
                    }
                    if (rs1 != 0) {
                        /* CSRRW - 写入 CSR */
                    }
                } else {
                    /* 其他 CSR 默认为 0 */
                    if (rd != 0) {
                        lane->gpr[rd] = 0;
                    }
                }
            } else if (funct3 == 0x2) { /* CSRRS (Read) */
                uint32_t csr_addr = (inst >> 20) & 0xFFF;
                
                if (csr_addr == CSR_MHARTID) {
                    if (rd != 0) {
                        lane->gpr[rd] = lane->mhartid;
                    }
                } else {
                    if (rd != 0) {
                        lane->gpr[rd] = 0;
                    }
                }
            } else if (funct3 == 0x0) {
                if ((inst & 0xFFF) == 0x000) { /* ECALL */
                    result->stop = true;
                }
            }
        }
        break;

    default:
        /* 未知指令 */
        break;
    }
}

/*
 * ============================================================================
 * Warp 初始化
 * ============================================================================
 */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));
    
    warp->warp_id = warp_id;
    warp->thread_id_base = thread_id_base;
    warp->active_mask = 0;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    for (uint32_t i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];
        
        lane->pc = pc;
        
        if (i < num_threads) {
            /* 编码 mhartid: block_id << 13 | warp_id << 5 | lane_id */
            lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
            lane->active = true;
            warp->active_mask |= (1 << i);
        } else {
            lane->active = false;
        }
        
        /* x0 永远是 0 */
        lane->gpr[0] = 0;
    }
}

/*
 * ============================================================================
 * Warp 执行 (SIMT - 锁步执行)
 * ============================================================================
 */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    ExecResult result = { .stop = false, .halted = false };
    uint32_t cycles = 0;

    while (!result.stop && cycles < max_cycles) {
        /* 锁步执行: 所有活跃 lane 执行同一条指令 */
        for (uint32_t i = 0; i < GPGPU_WARP_SIZE; i++) {
            exec_lane(s, &warp->lanes[i], &result);
        }
        cycles++;
        
        if (result.halted) {
            break;
        }
    }

    if (cycles >= max_cycles) {
        return -1;
    }
    return 0;
}

/*
 * ============================================================================
 * Kernel 执行
 * ============================================================================
 */
int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t kernel_addr = s->kernel.kernel_addr_lo; /* 低 32 位即可 */
    uint32_t block_threads = calc_block_threads(s);
    uint32_t threads_per_warp = GPGPU_WARP_SIZE;
    uint32_t num_warps = (block_threads + threads_per_warp - 1) / threads_per_warp;

    for (uint32_t bz = 0; bz < s->kernel.grid_dim[2]; bz++) {
        for (uint32_t by = 0; by < s->kernel.grid_dim[1]; by++) {
            for (uint32_t bx = 0; bx < s->kernel.grid_dim[0]; bx++) {
                uint32_t block_linear = block_id_to_linear(s, bx, by, bz);

                /* 遍历此 block 的所有 warps */
                for (uint32_t w = 0; w < num_warps; w++) {
                    GPGPUWarp warp;
                    uint32_t thread_base = w * threads_per_warp;
                    uint32_t threads_this_warp = threads_per_warp;
                    
                    if (thread_base + threads_this_warp > block_threads) {
                        threads_this_warp = block_threads - thread_base;
                    }

                    /* 初始化 warp */
                    memset(&warp, 0, sizeof(warp));
                    warp.warp_id = w;
                    warp.thread_id_base = thread_base;
                    warp.block_id[0] = bx;
                    warp.block_id[1] = by;
                    warp.block_id[2] = bz;

                    /* 初始化此 warp 的所有 lanes */
                    for (uint32_t lane = 0; lane < threads_per_warp; lane++) {
                        GPGPULane *l = &warp.lanes[lane];
                        l->pc = kernel_addr;
                        l->gpr[0] = 0;
                        
                        if (lane < threads_this_warp) {
                            /* 编码 mhartid */
                            l->mhartid = MHARTID_ENCODE(block_linear, w, lane);
                            l->active = true;
                            warp.active_mask |= (1 << lane);
                        } else {
                            l->active = false;
                        }
                    }

                    /* 执行 warp */
                    gpgpu_core_exec_warp(s, &warp, 1024);
                }
            }
        }
    }

    return 0;
}
