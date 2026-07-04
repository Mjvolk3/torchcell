---
id: j6frrlmtynkxnm73qqif3ok
title: Storage Memory Architecture
desc: ''
updated: 1783113391299
created: 1783112878109
---

> Companion to [[Hardware Inventory|gilahyper.hardware-inventory]] — that note is
> the durable per-part inventory (motherboard, CPU, memory, GPUs, exact drives and
> connectors); this note is the didactic *why* behind the storage/memory topology.

# GilaHyper Storage & Memory Architecture: A Didactic Reference

## 1. The Memory Hierarchy: First Principles

Every modern computer is built around a single uncomfortable truth: fast memory is expensive and small; cheap memory is slow and large. The engineering response is a **hierarchy** — multiple tiers of storage, each faster and smaller than the one below it, with the CPU at the top and magnetic tape at the bottom.[^1][^2]

### 1.1 The Canonical Hierarchy

| Tier              | Technology     | Latency     | Bandwidth    | Capacity (typical)   | Volatile? |
|-------------------|----------------|-------------|--------------|----------------------|-----------|
| CPU Register      | SRAM flip-flop | <1 ns       | ~TB/s        | Bytes                | Yes       |
| L1 Cache          | SRAM           | ~1 ns       | ~1 TB/s      | 32–64 KB / core      | Yes       |
| L2 Cache          | SRAM           | ~4 ns       | ~500 GB/s    | 256 KB – 1 MB / core | Yes       |
| L3 Cache (shared) | SRAM           | ~15–30 ns   | ~200 GB/s    | 8–256 MB             | Yes       |
| Main Memory       | DDR4/5 DRAM    | ~100 ns     | 50–200 GB/s  | GBs–TBs              | Yes       |
| NVMe SSD          | NAND Flash     | ~50–150 µs  | 5–15 GB/s    | TBs                  | No        |
| SATA SSD          | NAND Flash     | ~100–300 µs | 0.5–0.6 GB/s | TBs                  | No        |
| HDD               | Spinning Disk  | ~5–15 ms    | 0.1–0.3 GB/s | TBs–PBs              | No        |

[^2][^3][^4]

The key insight is the **orders-of-magnitude gaps**. Moving down one tier doesn't add 10% latency — it multiplies it by a factor of 10–1000. The gap from DRAM to NVMe SSD is roughly \(1000\times\). The gap from NVMe to HDD is another \(100\times\). From top to bottom, total span is about \(10^7\): L1 cache at 1 ns versus HDD at 10 ms.[^5][^3]

### 1.2 The Locality Principle

The hierarchy only works because of **locality of reference**: programs tend to access the same data repeatedly (temporal locality) and access data near what they just accessed (spatial locality). The CPU hardware automatically exploits this by promoting recently used data upward through the cache levels. When a program violates locality — for example, randomly sampling millions of small records scattered across a 20 TB dataset — it defeats the cache hierarchy entirely and the full latency of the bottom tier is paid on every access. This is exactly the pathology afflicting DL minibatch loading from spinning HDDs.[^6][^7]

***

## 2. Your CPU: The 5995WX and Its PCIe Fabric

The AMD Ryzen Threadripper PRO 5995WX is purpose-built for workstations that need massive IO bandwidth alongside CPU compute. Its distinguishing features relative to desktop CPUs:[^8][^9][^10][^11]

| Attribute                 | 5995WX    | Typical Desktop (Ryzen 9 7950X) |
|---------------------------|-----------|---------------------------------|
| Cores / Threads           | 64 / 128  | 16 / 32                         |
| L3 Cache                  | 256 MB    | 64 MB                           |
| Memory Channels           | 8         | 2                               |
| Max Memory Bandwidth      | ~200 GB/s | ~50 GB/s                        |
| CPU-direct PCIe 4.0 lanes | **128**   | 24                              |
| Socket                    | sWRX8     | AM5                             |

Those 128 CPU-direct PCIe 4.0 lanes are the defining feature for storage. In contrast, a typical desktop CPU has at most 24 PCIe lanes, and half are usually consumed by the GPU.

### 2.1 What "CPU-Direct" vs "Chipset" Means

A PCIe lane is a high-speed serial link — one pair of wires for sending, one for receiving. Data flows between the CPU and a device over these lanes at up to 16 GT/s per lane in PCIe 4.0, giving each lane roughly 2 GB/s of usable bandwidth in each direction.[^12]

**CPU-direct lanes** connect the device directly to the CPU's memory controller with no intermediary. Latency is minimal — just the physical wire delay plus controller overhead.[^13][^12]

**Chipset lanes** go through an extra hop: Device → PCIe link → AMD WRX80 Chipset → PCIe uplink → CPU. The uplink to the CPU is itself a shared bus (a fixed number of PCIe lanes, typically ×4–×8). When multiple chipset-attached devices are active simultaneously, they contend for this uplink.[^14][^12]

```
CPU (5995WX, 128 PCIe 4.0 lanes)
 ├─ ×16 Gen4 ──→ GPU 0 (RTX 6000 Ada)
 ├─ ×16 Gen4 ──→ GPU 1
 ├─ ×16 Gen4 ──→ GPU 2
 ├─ ×16 Gen4 ──→ GPU 3
 ├─ ×4 Gen4  ──→ M.2_1 (Solidigm P41 2TB, OS)
 ├─ ×4 Gen4  ──→ M.2_2 (OPEN ⭐ — target for SN850X)
 ├─ ×4 Gen3  ──→ U.2_1 → Micron 7400 PRO 7.68TB (/scratch)
 └─ ×? uplink──→ AMD WRX80 Chipset
                  ├─ ×4 Gen4 → M.2_3 (OPEN)
                  ├─ ×4 Gen3 → U.2_2 (OPEN, blocked if M.2_3 used)
                  ├─ 4× SATA → (chipset, unenumerated)
                  └─ PCIe  → ASMedia ASM1061/62
                               ├─ SATA → sdb (WD 26TB HDD)
                               └─ SATA → sdc (WD 26TB HDD)
```

***

## 3. Interfaces Compared: A Taxonomy

### 3.1 PCIe × NVMe × SATA — The Confusion Matrix

These terms are often conflated. A clean separation:

| Term          | What it describes                                               | Analogy             |
|---------------|-----------------------------------------------------------------|---------------------|
| **PCIe**      | The electrical bus / physical lane standard                     | The highway         |
| **NVMe**      | A protocol (command set) designed for flash storage over PCIe   | The traffic law     |
| **SATA**      | An older electrical bus + protocol, designed for spinning disks | A surface road      |
| **M.2**       | A physical connector shape (slot)                               | The on-ramp shape   |
| **U.2 / U.3** | A physical connector shape for 2.5" NVMe drives                 | A different on-ramp |

Your Micron 7400 PRO is a **NVMe drive** (protocol) running over **PCIe 4.0 ×4** (electrical bus), housed in a **2.5" U.3 enclosure** (physical form factor), connected via a **SFF-8639 cable** to the **U.2_1 port** on the motherboard.[^15]

Your WD Gold HDDs are **SATA** drives (both protocol and electrical bus), connected via **SATA III cables** (6 Gbps electrical standard) to the **ASMedia ASM1061 controllers**, which are themselves PCIe devices hanging off the chipset.

### 3.2 Performance by Interface

| Interface        | Random 4K IOPS       | Seq. Read BW       | Latency      | Typical use  |
|------------------|----------------------|--------------------|--------------|--------------|
| DDR4-3200 (DRAM) | — (byte-addressable) | ~50 GB/s / channel | ~100 ns      | Working data |
| PCIe 4.0 ×4 NVMe | ~1M IOPS             | ~7 GB/s            | ~50 µs       | Hot scratch  |
| PCIe 3.0 ×4 NVMe | ~500K IOPS           | ~3.5 GB/s          | ~70 µs       | Warm scratch |
| SATA III SSD     | ~90K IOPS            | ~550 MB/s          | ~100 µs      | Warm bulk    |
| SATA III HDD     | **~150 IOPS**        | ~250 MB/s          | **~5–10 ms** | Cold archive |

[^2][^3][^4]

The HDD row is the critical one. Note the IOPS: 150 versus 1,000,000. For a DL training loop that issues thousands of random small reads per batch (one read per sample, scattered across a large dataset), this is the difference between GPU utilization above 90% and GPU starvation below 10%.

***

## 4. Your Actual Hardware — All Tiers Mapped

| Tier              | Device                    | Capacity             | Interface         | Speed          | Role                                 |
|-------------------|---------------------------|----------------------|-------------------|----------------|--------------------------------------|
| GPU VRAM          | 4× RTX 6000 Ada           | 192 GB               | NVLink/PCIe       | ~2 TB/s        | Active model weights                 |
| DRAM              | 8× Samsung 64GB ECC RDIMM | **512 GB**           | 8-ch DDR4-3200    | ~200 GB/s      | CPU working set, pinned data buffers |
| Hot NVMe          | Micron 7400 PRO           | 7.68 TB              | PCIe 3.0 ×4 (U.3) | ~6.6 GB/s read | `/scratch` — hot dataset             |
| Hot NVMe (adding) | WD SN850X 8TB             | 8 TB                 | PCIe 4.0 ×4 (M.2) | ~7.3 GB/s read | M.2_2 or M.2_3 — overflow hot        |
| OS NVMe           | Solidigm P41 Plus         | 2 TB                 | PCIe 4.0 ×4 (M.2) | ~4 GB/s read   | OS, `/home`                          |
| Cold SATA         | 2× WD WD261KRYZ           | **2× 26 TB = 52 TB** | SATA III (HDD)    | ~250 MB/s      | Archive, datasets, backups           |

***

## 5. The "Large Memory System" Idea: Can We Fake a Bigger Memory?

This is the question you're circling — and it's a real systems research problem. The idea: use fast persistent storage (NVMe SSD or even HDD) as a **lower tier of "memory"** to create the illusion of a system with far more addressable fast storage than DRAM alone. Several concrete mechanisms exist, with very different performance profiles.

### 5.1 Principle: The Storage-Memory Continuum

\[
\text{Effective Memory} = \text{DRAM} + \text{NVMe}*\text{swap} + \text{HDD}*\text{cold}
\]

The hierarchy works when **access patterns have sufficient locality**: recently/frequently used pages stay in DRAM, less-used pages spill to NVMe, cold data stays on HDD. The system breaks down when access is **uniformly random** — then every tier must be consulted equally, and you pay the latency of the slowest tier on average.

### 5.2 Mechanism A: Linux Swap on NVMe (Kernel-Managed)

Linux swap allows the kernel to page memory out to a block device when DRAM pressure is high. With NVMe as the swap device, spilled pages come back at ~50–150 µs versus ~5–10 ms for HDD swap — a \(50–200\times\) improvement.[^2][^16]

**Setup on GilaHyper (conceptual):**

```bash
# Create a swapfile on /scratch (Micron NVMe)
fallocate -l 100G /scratch/swapfile
chmod 600 /scratch/swapfile
mkswap /scratch/swapfile
swapon /scratch/swapfile --priority=10

# Or a dedicated swap partition on the new SN850X
mkswap /dev/nvme2n1p1
swapon /dev/nvme2n1p1 --priority=10
```

**Practical limits:** Swap works for data that fits the page access model (4 KB pages, LRU eviction). For DL inference where you're loading model weights in large sequential chunks, NVMe swap can be surprisingly effective. For DL *training* with minibatch random reads, it doesn't help — the kernel can't predict which pages to keep warm.

### 5.3 Mechanism B: ZRAM (Compressed RAM — In-Memory Swap)

ZRAM creates a compressed swap device entirely within DRAM. Pages evicted from the hot working set are compressed (using zstd or lz4) and stored in a compressed pool in RAM itself. This trades CPU cycles for effective memory expansion.[^17][^18][^19]

\[
\text{Effective DRAM} \approx \text{Physical DRAM} \times \text{compression ratio}
\]

For model weights and activations (often float32 or bfloat16 tensors), compression ratios of 2–3× are achievable with zstd. On GilaHyper with 512 GB physical DRAM, ZRAM could theoretically give ~700–1000 GB effective addressable memory — at the cost of ~20–30% CPU overhead for compression/decompression.[^20][^17]

```bash
# Enable ZRAM on Rocky Linux 9
dnf install zram-generator
# Configure /etc/systemd/zram-generator.conf:
# [zram0]
# zram-size = ram / 2        # or a fixed size like 256G
# compression-algorithm = zstd
systemctl daemon-reload && systemctl start systemd-zram-setup@zram0.service
```

**For GilaHyper:** With 512 GB DRAM and 64 cores, ZRAM is a reasonable tool for expanding working set for inference (loading large models). For training, the CPU overhead may conflict with data preprocessing.

### 5.4 Mechanism C: mmap + DAX (Direct Access — NVMe as Byte-Addressable Memory)

Linux DAX (Direct Access) allows a filesystem on an NVMe drive to be `mmap`'d directly into process virtual address space, **bypassing the page cache**. The OS treats NVMe pages as an extension of virtual memory, using the CPU's virtual memory hardware (MMU) to map NVMe addresses alongside DRAM addresses.[^21][^22][^23]

\[
\text{Virtual Address Space} = \text{DRAM pages} \cup \text{NVMe pages (DAX)}
\]

This is what Intel Optane Persistent Memory (PMEM) was designed for — it sat on the DIMM slot and presented NVMe-class persistent storage at DRAM latency. Without Optane, you can approximate it with a DAX-capable NVMe filesystem (`ext4` with `-O dax` or `xfs` with `dax=always`).[^24][^25]

**The catch:** NVMe DAX latency is still ~50–150 µs per random access, versus ~100 ns for DRAM. For *sequential* access patterns (loading a model checkpoint, streaming a shard), this is acceptable. For *random byte-level access* (attention patterns, embedding lookups), the \(1000\times\) latency gap is still felt.

### 5.5 Mechanism D: Tiered Storage with a Staging Daemon (What TAIGA Does)

NCSA's **Taiga** system at UIUC is exactly the architecture you referenced. It implements a **hybrid NVMe + HDD tiered filesystem** using DDN EXAScaler / Lustre:[^26][^27][^28]

| Component     | Taiga (UIUC)                 | GilaHyper (equivalent)       |
|---------------|------------------------------|------------------------------|
| Filesystem    | Lustre (DDN EXAScaler)       | ext4 / xfs                   |
| Hot tier      | NVMe (100 TB usable per SSU) | 7.68 TB Micron + 8 TB SN850X |
| Cold tier     | HDD (5–9 PB usable per SSU)  | 52 TB (2× 26TB WD Gold)      |
| Tiering logic | DDN Stratagem policy engine  | Manual / dm-cache / custom   |
| Aggregate BW  | ~30–55 GB/s per SSU          | ~14 GB/s NVMe, ~0.5 GB/s HDD |
| Scale         | Petabytes                    | ~60 TB                       |

[^27][^26]

Taiga's key innovation is **automatic data promotion/demotion**: data accessed frequently is automatically migrated from HDD to NVMe; data that cools off is demoted back. On GilaHyper, you can approximate this with:

1. **dm-cache / LVM cache**: The kernel `dm-cache` target places the NVMe in front of the HDD as a block-level cache. Frequently read blocks are promoted to NVMe automatically, without application involvement.
2. **Application-level staging**: A background `rsync` or custom daemon moves the next epoch's data to NVMe while training on the current epoch (double-buffering).
3. **Streaming shards (MosaicML Streaming / WebDataset)**: Treats the HDD as a sequential source, NVMe as an LRU cache of recent shards. Converts random access into sequential reads.

***

## 6. The 26 TB HDDs as Cold Storage — What's Realistic

The direct question: **can the 52 TB of spinning HDD act as a large virtual memory extension for DL training?**

The honest answer is: **no, not directly for training; yes, for staging and checkpointing.**

### 6.1 Why HDDs Fail for Random DL IO

A DL training epoch over a 20 TB dataset with 32-sample minibatches issues roughly:

\[
\text{IO ops} = \frac{20 \times 10^{12} \text{ bytes}}{32 \times \bar{S}_\text{sample}}
\]

where \(\bar{S}_\text{sample}\) is average sample size. For 1 MB samples, that's ~600,000 IO operations per epoch. A single HDD at 150 IOPS takes:

\[
t_\text{HDD} = \frac{600{,}000}{150} \approx 4{,}000 \text{ seconds} \approx 67 \text{ minutes of pure IO per epoch}
\]

Four RTX 6000 Ada GPUs, if data-fed, can process that same epoch in perhaps 10–15 minutes of compute. The HDD would create a \(4–7\times\) IO bottleneck: GPUs idle >80% of the time waiting for data.[^2][^3]

Striping the two HDDs in RAID 0 helps only on sequential throughput (~500 MB/s combined), not IOPS (still ~300 IOPS total). The rotational latency per seek (~5 ms) is physical and irreducible.

### 6.2 What HDDs Are Good For

| Use case                                  | HDD suitable?                              | Why                                       |
|-------------------------------------------|--------------------------------------------|-------------------------------------------|
| Dataset archive (write once, rarely read) | ✅ Yes                                      | Sequential write at 250 MB/s is fine      |
| Model checkpoint storage                  | ✅ Yes                                      | Sequential write, infrequent              |
| Staging datasets to NVMe overnight        | ✅ Yes                                      | Slow but background, not on critical path |
| Backup                                    | ✅ Yes                                      | Sequential, time-insensitive              |
| Active training data source               | ❌ No                                       | IOPS and seek latency are prohibitive     |
| Swap / overflow memory                    | ⚠️ No for training; marginal for inference | Seek latency destroys random access       |

### 6.3 The Practical Tiered Strategy for GilaHyper

```
GPU VRAM (192 GB) ← active batch, model weights
      ↑ PCIe DMA
DRAM (512 GB) ← Python process, DataLoader workers, pinned buffers
      ↑ CPU loads
NVMe Scratch (7.68 TB + 8 TB = 15.68 TB) ← HOT: active training dataset, current run
      ↑ background rsync / streaming shard cache
HDD (52 TB) ← COLD: full dataset archive, past checkpoints, raw data
```

The key engineering move is to **keep the GPU's data path entirely within NVMe** and use the HDD only for background staging. A concrete implementation:

```bash
# LVM stripe across Micron + SN850X for 15+ TB hot tier
pvcreate /dev/nvme0n1 /dev/nvme2n1
vgcreate vg_hot /dev/nvme0n1 /dev/nvme2n1
lvcreate -l 100%VG --stripes 2 --stripesize 64K -n scratch vg_hot
mkfs.ext4 /dev/vg_hot/scratch
mount /dev/vg_hot/scratch /scratch

# Background staging from HDD to hot NVMe
rsync -av --progress /cold/dataset/epoch_N/ /scratch/dataset/ &
# Train on /scratch while rsync runs
```

### 6.4 Striped NVMe: The Math

With two NVMe drives striped (RAID 0 / LVM stripe), each ×4 PCIe 4.0, theoretical peak sequential read is:

\[
BW_\text{stripe} = BW_1 + BW_2 \approx 6.6 + 7.3 = 13.9 \text{ GB/s}
\]

At this speed, reading 20 TB of data sequentially takes:

\[
t = \frac{20 \times 10^{12}}{13.9 \times 10^9} \approx 1{,}440 \text{ seconds} \approx 24 \text{ minutes}
\]

Compare to GPU compute time per epoch (~15 min): IO is now within 1.6× of compute, and with effective prefetching (PyTorch DataLoader with `num_workers ≥ 4`, `prefetch_factor ≥ 2`), the GPU never needs to wait.[^2]

***

## 7. Summary: Principles for the GilaHyper Architecture

1. **The memory hierarchy is immutable physics.** Latency gaps between tiers are not engineering choices — they are consequences of how electrons and photons move and how NAND cells charge. Design around them, not against them.

2. **CPU-direct PCIe lanes are always preferable for latency-sensitive IO.** Your 5995WX has 128 — use them. M.2_2 on the CPU is faster than M.2_3 on the chipset; U.2_1 on the CPU is faster than U.2_2 on the chipset.[^12][^13]

3. **SATA HDDs are sequential beasts.** Their 250 MB/s sequential throughput is fine for staging; their 150 IOPS random performance is fatal for training. Never place active training data on them.

4. **NVMe is the right hot tier.** At 1M IOPS and 7 GB/s, the Micron + SN850X stripe can feed four high-end GPUs without bottleneck — if data is organized for sequential shard access.

5. **Large "virtual memory" from HDDs is possible but workload-constrained.** dm-cache can automate block-level tiering between NVMe and HDD. Streaming loaders (MosaicML Streaming, WebDataset) can make HDD the source for sequential shard reads. Neither approach makes HDD random access fast — they avoid it entirely.

6. **Taiga (UIUC) works at petabyte scale** using the same principle: NVMe as a hot cache tier in front of HDDs, with policy engines managing promotion/demotion automatically on a Lustre filesystem. GilaHyper is a workstation-scale equivalent of the same architecture.[^26][^27][^28]

7. **Your 512 GB DRAM is a large asset.** At 8-channel DDR4-3200, it delivers ~200 GB/s to the CPU — more bandwidth than any NVMe tier. For inference, loading a quantized 70B model (~35 GB) fits entirely in DRAM with room for KV caches. ZRAM can extend this further at the cost of CPU cycles.[^17][^20]

---

## References

1. [Memory hierarchy - Wikipedia](https://en.wikipedia.org/wiki/Memory_hierarchy)

2. [Storage](https://www.cs.cornell.edu/courses/cs3410/2019sp/schedule/slides/17-storage-notes.pdf)

3. [Hot Warm Cold Storage: Tier Guide - TechCompare](https://www.techcompare.app/latency-visualizer/storage-tier-guide) - Hot storage is NVMe SSD for active data. Warm is HDD or object storage. Cold is archival tape or deep archive.

4. [Understanding Memory Hierarchy for Optimal System Performance](https://www.linkedin.com/posts/cmabuzar_mac-it-muhammad-activity-7424705289263722496-VQFl)

5. [Memory Hierarchy Primer — Algorhythm](https://algo-rhythm.dev/en/memory-hierarchy/)

6. [Storage Hierarchy, Caching, and Locality](https://www.cs.princeton.edu/courses/archive/spr21/cos217/lectures/16_StorageHierarchy.pdf)

7. [Data Management Systems](https://ethz.ch/content/dam/ethz/special-interest/infk/inst-cp/inst-cp-dam/education/courses/2020-fall/data-management-systems/slides/DMS-HS20-Storage_Memory_Hierarchy.pdf)

8. [AMD Ryzen Threadripper PRO 5995WX - CpuTronic.com](https://cputronic.com/cpu/amd-ryzen-threadripper-pro-5995wx)

9. [Specifications for AMD Threadripper PRO 5995WX - Hashrate](https://www.hashrate.no/cpus/5995wx/specs)

10. [AMD Ryzen Threadripper Pro 5995WX CPU 64 Core 128 Thread 2.7/4.5GHz](https://www.punchtechnology.co.uk/product/amd-ryzen-threadripper-pro-5995wx-cpu-64-core-128-thread-2-7-4-5ghz/)

11. [AMD Ryzen Threadripper PRO WX-5000 Series QRG](https://www.amd.com/content/dam/amd/en/documents/partner-hub/threadripper/ryzen-threadripper-pro-wx-5000-series-qrg-generational.pdf)

12. [CPU Vs Chipset Pcie Lanes - UMA Technology](https://umatechnology.org/cpu-vs-chipset-pcie-lanes/)

13. [Physical Medium — AMD Docs](https://docs.amd.com/r/en-US/ug1523-x3522-user/Other-Considerations)

14. [The problem with PCIe is not bandwidth, it is the limit in lanes](https://news.ycombinator.com/item?id=29897701)

15. [Micron® 7400 PRO 7.68TB NVMe™ U.3 (7mm) Non-SED - Crucial](https://www.crucial.com/ssd/7400_pro/mtfdkcb7t6tdz-2az18abyyr)

16. [32 GB RAM + 7000MB/s NVME: Proper swap setup?](https://bbs.archlinux.org/viewtopic.php?id=283971)

17. [Something to consider trying if you almost have enough RAM (r/LocalLLaMA)](https://www.reddit.com/r/LocalLLaMA/comments/1hwix4d/something_to_consider_trying_if_you_almost_have/)

18. [zram - ArchWiki](https://wiki.archlinux.org/title/Zram)

19. [zswap — The Linux Kernel documentation](https://www.kernel.org/doc/html/v4.18/vm/zswap.html)

20. [Debunking zswap and zram myths - Chris Down](https://chrisdown.name/2026/03/24/zswap-vs-zram-when-to-use-what.html)

21. [SUSE Labs 2019 - hmmap: How to Combine DAX and DRAM Caching](https://www.youtube.com/watch?v=RyeEE8OsKIw)

22. [User Directed Tiered Memory Management](https://www.usenix.org/conference/linuxfastsummit18/presentation/manzanares)

23. [I/O Approaches in Modern Storage Systems - CS647](https://www.csd.uoc.gr/~hy647/lectures/09_iopath.pdf)

24. [Buffer Management with NVM - cs.wisc.edu](https://pages.cs.wisc.edu/~yxy/cs764-f21/slides/L5.pdf)

25. [Towards Hybrid Storage Devices with Block and DAX](https://www.betriebssysteme.org/wp-content/uploads/2024/03/4.1-hybrid-storage.pdf)

26. [System Architecture — UIUC NCSA Taiga User Guide](https://docs.ncsa.illinois.edu/systems/taiga/en/latest/user-guide/architecture.html)

27. [Storage Environment - Illinois Campus Cluster Program](https://campuscluster.illinois.edu/about/system-info/storage-environment/)

28. [UIUC NCSA Taiga User Guide](https://docs.ncsa.illinois.edu/systems/taiga/en/latest/index.html)
