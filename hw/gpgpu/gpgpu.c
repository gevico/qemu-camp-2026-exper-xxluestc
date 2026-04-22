/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"
#include "gpgpu_core.h"

/* 前向声明 */
static void gpgpu_execute_kernel(GPGPUState *s);

/* TODO: Implement MMIO control register read */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque; 
    uint32_t val = 0;

    switch (addr) {
    case GPGPU_REG_DEV_ID:
        val = GPGPU_DEV_ID_VALUE;
        break;
    case GPGPU_REG_DEV_VERSION:
        val = GPGPU_DEV_VERSION_VALUE;
        break;
    case GPGPU_REG_VRAM_SIZE_LO:   // 0x000C
        val = (uint32_t)(s->vram_size & 0xFFFFFFFF);
        break;
    case GPGPU_REG_VRAM_SIZE_HI:   // 0x0010
        val = (uint32_t)(s->vram_size >> 32);
        break;
    case GPGPU_REG_GLOBAL_STATUS:
        val = s->global_status;  // 返回 s->global_status (初始化为 GPGPU_STATUS_READY)
        break;
    case GPGPU_REG_GLOBAL_CTRL:
        val = s->global_ctrl;
        break;
    case GPGPU_REG_GRID_DIM_X:
        val = s->grid_dim_x;
        break;
    case GPGPU_REG_GRID_DIM_Y:
        val = s->grid_dim_y;
        break;
    case GPGPU_REG_GRID_DIM_Z:
        val = s->grid_dim_z;
        break;
    case GPGPU_REG_BLOCK_DIM_X:
        val = s->block_dim_x;
        break;
    case GPGPU_REG_BLOCK_DIM_Y:
        val = s->block_dim_y;
        break;
    case GPGPU_REG_BLOCK_DIM_Z:
        val = s->block_dim_z;
        break;
    case GPGPU_REG_IRQ_ENABLE:
        val = s->irq_enable;
        break;
    case GPGPU_REG_IRQ_STATUS:
        val = s->irq_status;
        break;
    case GPGPU_REG_DMA_SRC_LO:
        val = (uint32_t)(s->dma.src_addr & 0xFFFFFFFF);
        break;
    case GPGPU_REG_DMA_SRC_HI:
        val = (uint32_t)(s->dma.src_addr >> 32);
        break;
    case GPGPU_REG_DMA_DST_LO:
        val = (uint32_t)(s->dma.dst_addr & 0xFFFFFFFF);
        break;
    case GPGPU_REG_DMA_DST_HI:
        val = (uint32_t)(s->dma.dst_addr >> 32);
        break;
    case GPGPU_REG_DMA_SIZE:
        val = s->dma.size;
        break;
    case GPGPU_REG_DMA_CTRL:
        val = s->dma.ctrl;
        break;
    case GPGPU_REG_DMA_STATUS:
        val = s->dma.status;
        break;
    case GPGPU_REG_THREAD_ID_X:
        val = s->simt.thread_id[0];
        break;
    case GPGPU_REG_THREAD_ID_Y:
        val = s->simt.thread_id[1];
        break;
    case GPGPU_REG_THREAD_ID_Z:
        val = s->simt.thread_id[2];
        break;
    case GPGPU_REG_BLOCK_ID_X:
        val = s->simt.block_id[0];
        break;
    case GPGPU_REG_BLOCK_ID_Y:
        val = s->simt.block_id[1];
        break;
    case GPGPU_REG_BLOCK_ID_Z:
        val = s->simt.block_id[2];
        break;
    case GPGPU_REG_WARP_ID:
        val = s->simt.warp_id;
        break;
    case GPGPU_REG_LANE_ID:
        val = s->simt.lane_id;
        break;
    case GPGPU_REG_THREAD_MASK:
        val = s->simt.thread_mask;
        break;
    case GPGPU_REG_KERNEL_ADDR_LO:
        val = s->kernel.kernel_addr_lo;
        break;
    case GPGPU_REG_KERNEL_ADDR_HI:
        val = s->kernel.kernel_addr_hi;
        break;
    default:
        val = 0;
        break;
    }
    return val;
}

/* MMIO control register write */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;

    switch (addr) {
    case GPGPU_REG_GLOBAL_CTRL:
        s->global_ctrl = (uint32_t)val;
        if (val & GPGPU_CTRL_RESET) {
            memset(&s->simt, 0, sizeof(s->simt));
        }
        break;
    case GPGPU_REG_GRID_DIM_X:
        s->grid_dim_x = (uint32_t)val;
        break;
    case GPGPU_REG_GRID_DIM_Y:
        s->grid_dim_y = (uint32_t)val;
        break;
    case GPGPU_REG_GRID_DIM_Z:
        s->grid_dim_z = (uint32_t)val;
        break;
    case GPGPU_REG_BLOCK_DIM_X:
        s->block_dim_x = (uint32_t)val;
        break;
    case GPGPU_REG_BLOCK_DIM_Y:
        s->block_dim_y = (uint32_t)val;
        break;
    case GPGPU_REG_BLOCK_DIM_Z:
        s->block_dim_z = (uint32_t)val;
        break;
    case GPGPU_REG_IRQ_ENABLE:
        s->irq_enable = (uint32_t)val;
        break;
    case GPGPU_REG_DMA_SRC_LO:
        s->dma.src_addr = (s->dma.src_addr & ~0xFFFFFFFFULL) | (uint32_t)val;
        break;
    case GPGPU_REG_DMA_SRC_HI:
        s->dma.src_addr = (s->dma.src_addr & 0xFFFFFFFFULL) | ((uint64_t)(uint32_t)val << 32);
        break;
    case GPGPU_REG_DMA_DST_LO:
        s->dma.dst_addr = (s->dma.dst_addr & ~0xFFFFFFFFULL) | (uint32_t)val;
        break;
    case GPGPU_REG_DMA_DST_HI:
        s->dma.dst_addr = (s->dma.dst_addr & 0xFFFFFFFFULL) | ((uint64_t)(uint32_t)val << 32);
        break;
    case GPGPU_REG_DMA_SIZE:
        s->dma.size = (uint32_t)val;
        break;
    case GPGPU_REG_DMA_CTRL:
        s->dma.ctrl = (uint32_t)val;
        break;
    case GPGPU_REG_DMA_STATUS:
        s->dma.status = (uint32_t)val;
        break;
    case GPGPU_REG_THREAD_ID_X:
        s->simt.thread_id[0] = (uint32_t)val;
        break;
    case GPGPU_REG_THREAD_ID_Y:
        s->simt.thread_id[1] = (uint32_t)val;
        break;
    case GPGPU_REG_THREAD_ID_Z:
        s->simt.thread_id[2] = (uint32_t)val;
        break;
    case GPGPU_REG_BLOCK_ID_X:
        s->simt.block_id[0] = (uint32_t)val;
        break;
    case GPGPU_REG_BLOCK_ID_Y:
        s->simt.block_id[1] = (uint32_t)val;
        break;
    case GPGPU_REG_BLOCK_ID_Z:
        s->simt.block_id[2] = (uint32_t)val;
        break;
    case GPGPU_REG_WARP_ID:
        s->simt.warp_id = (uint32_t)val;
        break;
    case GPGPU_REG_LANE_ID:
        s->simt.lane_id = (uint32_t)val;
        break;
    case GPGPU_REG_THREAD_MASK:
        s->simt.thread_mask = (uint32_t)val;
        break;
    case GPGPU_REG_KERNEL_ADDR_LO:
        s->kernel.kernel_addr_lo = (uint32_t)val;
        break;
    case GPGPU_REG_KERNEL_ADDR_HI:
        s->kernel.kernel_addr_hi = (uint32_t)val;
        break;
    case GPGPU_REG_DISPATCH:
        gpgpu_execute_kernel(s);
        break;
    default:
        break;
    }
}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* VRAM read */
static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque;
    uint64_t val = 0;

    if (addr + size > s->vram_size) {
        return 0;
    }

    switch (size) {
    case 1:
        val = s->vram_ptr[addr];
        break;
    case 2:
        val = *(uint16_t *)(s->vram_ptr + addr);
        break;
    case 4:
        val = *(uint32_t *)(s->vram_ptr + addr);
        break;
    case 8:
        val = *(uint64_t *)(s->vram_ptr + addr);
        break;
    default:
        val = 0;
        break;
    }
    return val;
}

/* VRAM write */
static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;

    if (addr + size > s->vram_size) {
        return;
    }

    switch (size) {
    case 1:
        s->vram_ptr[addr] = (uint8_t)val;
        break;
    case 2:
        *(uint16_t *)(s->vram_ptr + addr) = (uint16_t)val;
        break;
    case 4:
        *(uint32_t *)(s->vram_ptr + addr) = (uint32_t)val;
        break;
    case 8:
        *(uint64_t *)(s->vram_ptr + addr) = (uint64_t)val;
        break;
    default:
        break;
    }
}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)size;
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)val;
    (void)size;
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* gpgpu_execute_kernel - 触发内核执行 */
static void gpgpu_execute_kernel(GPGPUState *s)
{
    /* 更新 kernel 结构中的地址（使用已有的 kernel_addr_lo/hi） */
    
    /* 更新 grid/block 维度 */
    s->kernel.grid_dim[0] = s->grid_dim_x;
    s->kernel.grid_dim[1] = s->grid_dim_y;
    s->kernel.grid_dim[2] = s->grid_dim_z;
    s->kernel.block_dim[0] = s->block_dim_x;
    s->kernel.block_dim[1] = s->block_dim_y;
    s->kernel.block_dim[2] = s->block_dim_z;

    /* 标记设备忙 */
    s->global_status = GPGPU_STATUS_BUSY;

    /* 调用核心执行引擎 */
    gpgpu_core_exec_kernel(s);

    /* 执行完成，标记设备就绪 */
    s->global_status = GPGPU_STATUS_READY;
}

/* TODO: Implement DMA completion handler */
static void gpgpu_dma_complete(void *opaque)
{
    (void)opaque;
}

/* TODO: Implement kernel completion handler */
static void gpgpu_kernel_complete(void *opaque)
{
    (void)opaque;
}

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr) {
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }

    /* BAR 0: control registers */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: VRAM */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: doorbell registers */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0, 0xFE000,
                  &s->ctrl_mmio, 0, 0xFF000,
                  0, errp)) {
        g_free(s->vram_ptr);
        return;
    }

    msi_init(pdev, 0, 1, true, false, errp);

    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);
    s->kernel_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,
                                   gpgpu_kernel_complete, s);

    s->global_status = GPGPU_STATUS_READY;
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    timer_free(s->dma_timer);
    timer_free(s->kernel_timer);
    g_free(s->vram_ptr);
    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));
    memset(&s->simt, 0, sizeof(s->simt));
    timer_del(s->dma_timer);
    timer_del(s->kernel_timer);
    if (s->vram_ptr) {
        memset(s->vram_ptr, 0, s->vram_size);
    }
}

static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()
    }
};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { }
    },
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)
