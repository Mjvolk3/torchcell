---
id: obs5hvd6szslwz8aylja2qv
title: Hardware Inventory
desc: ''
updated: 1781033990640
created: 1781033990640
---

# GilaHyper — Hardware Inventory

Durable reference for the `gilahyper` workstation, captured 2026-04-20 from the live system (originally prepared for an expansion discussion with Exxact). Contact/serial/shipping details from the original capture are intentionally omitted from this committed note.

Planned additions under consideration: an additional NVMe in the free onboard M.2 slot, and two large internal HDDs (one data volume, one backups). GPUs remain in place; memory is not being changed.

## System / OS

| Field    | Value                        |
|----------|------------------------------|
| Hostname | gilahyper                    |
| OS       | Rocky Linux 9.5 (Blue Onyx)  |
| Kernel   | 5.14.0-503.35.1.el9_5.x86_64 |
| Arch     | x86_64                       |
| BIOS     | AMI 1602, 2024-09-04         |

## Motherboard

| Field   | Value                          |
|---------|--------------------------------|
| Vendor  | ASUSTeK                        |
| Model   | **Pro WS WRX80E-SAGE SE WIFI** |
| Rev     | 1.xx                           |
| Chipset | AMD WRX80                      |
| Socket  | sWRX8                          |

## CPU

| Field           | Value                                 |
|-----------------|---------------------------------------|
| Model           | AMD **Ryzen Threadripper PRO 5995WX** |
| Cores / Threads | 64 / 128                              |
| Sockets         | 1                                     |
| L3 cache        | 256 MiB                               |

## Memory (installed)

Verified via `sudo dmidecode -t memory` — all 8 channels populated, fully matched kit.

| Field               | Value                                     |
|---------------------|-------------------------------------------|
| Installed           | **8 × 64 GB = 512 GB**                    |
| Slots               | 8 populated / 8 total                     |
| Type                | DDR4 **RDIMM** (Registered/Buffered), ECC |
| Speed (rated / configured) | 3200 MT/s / 3200 MT/s              |
| Rank                | 2R (dual-rank)                            |
| Voltage             | 1.2 V                                     |
| Manufacturer        | Samsung                                   |
| Part Number         | **M393A8G40AB2-CWE** (all 8 identical)    |
| Channels populated  | P0 CHANNEL A – H (all 8)                  |
| ECC                 | Multi-bit ECC (72-bit total / 64-bit data) |

## GPUs

4× NVIDIA **RTX 6000 Ada Generation** (AD102GL, 48 GB each, 192 GB total VRAM). Driver 580.82.07, CUDA 13.0.

| GPU | PCIe Bus ID | VRAM  |
|-----|-------------|-------|
| 0   | 01:00.0     | 48 GB |
| 1   | 2B:00.0     | 48 GB |
| 2   | 41:00.0     | 48 GB |
| 3   | 61:00.0     | 48 GB |

All four are double-slot cards; they occupy four PCIe 4.0 x16 slots and physically block the slots between them. In practice no full-height PCIe slot remains available for add-in cards while the GPUs are installed.

## Storage — NVMe (populated)

| Device    | Model                                            | Size    | PCIe Addr | Mount / LVM                                            |
|-----------|--------------------------------------------------|---------|-----------|--------------------------------------------------------|
| `nvme0n1` | Micron **7400 PRO** (MTFDKCB7T6TDZ)              | 7.68 TB | 23:00.0   | `/scratch` (ext4, single partition)                    |
| `nvme1n1` | Solidigm **P41 Plus** (SSDPFKNU020TZ, DRAM-less) | 2 TB    | 2a:00.0   | `/boot/efi`, `/boot`, LVM VG `rl` → `/`, swap, `/home` |

Two M.2 slots populated. Board has 3 onboard M.2 slots total → **1 onboard M.2 slot available**.

## Storage — SATA

| Controller         | PCIe Addr | Ports | In use |
|--------------------|-----------|-------|--------|
| ASMedia ASM1061/62 | 27:00.0   | 2     | 0      |
| ASMedia ASM1061/62 | 28:00.0   | 2     | 0      |

4 SATA ports currently enumerated, all empty. Board spec lists additional chipset SATA via SlimSAS — not enumerated by the running kernel; confirm whether a SlimSAS breakout or BIOS option is available on this build.

## Filesystems (live `df`, 2026-04-20)

| Mount       | FS   | Size  | Used  | Use% |
|-------------|------|-------|-------|------|
| `/`         | xfs  | 70 G  | 61 G  | 87%  |
| `/home`     | xfs  | 1.8 T | 465 G | 26%  |
| `/boot`     | xfs  | 960 M | 530 M | 56%  |
| `/boot/efi` | vfat | 599 M | 7 M   | 2%   |
| `/scratch`  | ext4 | 7.0 T | 4.6 T | 69%  |

## Summary of expansion headroom

| Interface           | Free                                                                |
|---------------------|---------------------------------------------------------------------|
| Onboard M.2         | 1 slot                                                              |
| SATA                | 4 ports (ASMedia); possibly more via chipset / SlimSAS — to confirm |
| PCIe x16 (physical) | 0 usable with 4 GPUs installed                                      |

## Open hardware questions

1. **NVMe:** any issue adding a 7.68 TB enterprise M.2 NVMe in the remaining onboard M.2 slot (2280 or 22110)? Which form factor does the open slot support?
2. **SATA / HDDs:** is a chipset SATA breakout (SlimSAS → 4× SATA) available on this chassis, or are we limited to the 4 ASMedia ports? Goal is two large CMR HDDs (one working data, one backup).
3. **Chassis / PSU:** how many free 3.5" bays and SATA power leads are available? Any constraint for adding 2 HDDs.
