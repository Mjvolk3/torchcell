| job | round | arm | commit | kg_config | cpus | mem_gb | inprocess_max_records | single_pass | fast_writer | io_to_total_worker_ratio | single_pass_chunk_budget_mb | chunks_per_worker | completion_order | inprocess_max_mb | wall_s | slurm_elapsed_s | peak_mem_gb | mean_cores | rows | csv_bytes | state | other_overrides |
|--:|---|---|---|---|--:|--:|---|---|---|---|---|---|---|---|--:|--:|--:|--:|--:|--:|---|---|
| 2874 | r4 | profile-writer | f5110338 |  | 16 | 64 | 25000 | true |  |  |  |  |  |  | 316.5 | 381 | 42.6 | 5.7 |  |  | FAILED | datasets=[DmfCostanzo2016Dataset] subset.per_dataset.DmfCostanzo2016Dataset=500000 |
| 2884 | r5 | full-51-32cpu | 1fdaaa18 |  | 32 | 128 | 25000 | true | true |  |  |  |  |  |  | 66 |  |  |  |  | FAILED |  |
| 2905 | r5 | full-51-32cpu-b | e48ffe2f |  | 32 | 128 | 25000 | true | true | 0.02 |  |  |  |  |  | 28,679 | 128.0 | 8.4 |  |  | FAILED |  |
| 2918 | r7 | b48-g2 | be51e6d1 |  | 32 | 128 | 25000 | true | true | 0.02 | 48 | 2 |  |  |  | 132 |  |  |  |  | FAILED |  |
| 2919 | r7 | b128-g2 | be51e6d1 |  | 32 | 128 | 25000 | true | true | 0.02 | 128 | 2 |  |  |  | 118 |  |  |  |  | FAILED |  |
| 2920 | r7 | b48-g8 | be51e6d1 |  | 32 | 128 | 25000 | true | true | 0.02 | 48 | 8 |  |  |  | 120 |  |  |  |  | FAILED |  |
| 2921 | r7 | b128-g8 | be51e6d1 |  | 32 | 128 | 25000 | true | true | 0.02 | 128 | 8 |  |  |  | 122 |  |  |  |  | FAILED |  |
| 2936 | r8 | full-51-shared-interned | b3f95cae |  | 32 | 128 | 25000 | true | true | 0.02 | 128 |  |  |  |  | 16,354 | 128.0 | 11.5 |  |  | FAILED |  |
| 2995 | r10 | g8 | a0b30e3b |  | 24 | 96 | 25000 | true | true | 0.02 | 128 | 8 |  |  |  | 149 |  |  |  |  | FAILED |  |
| 2996 | r10 | order-g8 | a0b30e3b |  | 24 | 96 | 25000 | true | true | 0.02 | 128 | 8 | true |  |  | 131 |  |  |  |  | FAILED |  |
| 2935 | r8 | bloom-shared-interned | dffbe79d | kg_bench_bloom | 32 | 128 | 25000 | true | true | 0.02 | 48 |  |  |  | 147.4 | 242 | 25.9 | 17.2 | 4230530 | 12910946619 | COMPLETED |  |
| 2934 | r8 | bloom-alone | c3f6b940 | kg_bench_bloom | 32 | 128 | 25000 | true | true | 0.02 | 48 |  |  |  | 834.7 | 1,131 | 83.9 | 4.5 | 4230530 | 12910969040 | COMPLETED |  |
| 2998 | r10 | expr-bytes | a0b30e3b | kg_bench_expr | 16 | 64 | 25000 | true | true | 0.02 | 128 |  |  | 256 | 974.3 | 1,031 | 43.6 | 2.7 | 173100 | 12658295605 | COMPLETED |  |
| 2997 | r10 | expr-inproc | a0b30e3b | kg_bench_expr | 16 | 64 | 25000 | true | true | 0.02 | 128 |  |  |  | 2,320.1 | 2,382 | 26.3 | 1.0 | 173100 | 12658347316 | COMPLETED |  |
| 3070 | r11 | mem60-48 | 33e0c048 | kg_bench_ladder | 48 | 192 | 25000 | true | true | 0.02 | 128 | 64 | true |  | 582.1 | 644 | 141.5 | 17.6 | 29736985 | 13431710999 | COMPLETED | adapters.pool_memory_fraction=0.6 |
| 2881 | r5 | fast-writer | 60b09ac6 | kg_bench_ladder | 48 | 192 | 25000 | true | true |  |  |  |  |  | 591.3 | 660 | 115.0 | 16.0 | 29736977 | 39275464550 | COMPLETED |  |
| 2882 | r5 | fast-writer-io0.02 | 60b09ac6 | kg_bench_ladder | 48 | 192 | 25000 | true | true | 0.02 |  |  |  |  | 596.3 | 660 | 124.5 | 15.7 | 29736977 | 39275464647 | COMPLETED |  |
| 3069 | r11 | g2-48 | 33e0c048 | kg_bench_ladder | 48 | 192 | 25000 | true | true | 0.02 | 128 | 2 | true |  | 604.2 | 669 | 104.1 | 16.9 | 29736985 | 13431711716 | COMPLETED |  |
| 3068 | r11 | mem60-24 | 33e0c048 | kg_bench_ladder | 24 | 96 | 25000 | true | true | 0.02 | 128 | 64 | true |  | 641.6 | 708 | 63.3 | 14.2 | 29736985 | 13431713158 | COMPLETED | adapters.pool_memory_fraction=0.6 |
| 3066 | r10 | order-slim | 4f399ff4 | kg_bench_ladder | 24 | 96 | 25000 | true | true | 0.02 | 128 | 2 | true |  | 643.8 | 704 | 62.3 | 13.3 | 29736985 | 13431713136 | COMPLETED |  |
| 2931 | r7 | b128-g2 | 98797536 | kg_bench_ladder | 32 | 128 | 25000 | true | true | 0.02 | 128 | 2 |  |  | 652.8 | 748 | 93.5 | 14.0 | 29736977 | 39275466626 | COMPLETED |  |
| 2883 | r5 | fast-writer-32cpu | 193d9761 | kg_bench_ladder | 32 | 128 | 25000 | true | true |  |  |  |  |  | 723.6 | 790 | 92.3 | 14.8 | 29736977 | 39275469188 | COMPLETED |  |
| 2994 | r10 | order | a0b30e3b | kg_bench_ladder | 24 | 96 | 25000 | true | true | 0.02 | 128 | 2 | true |  | 752.4 | 817 | 67.8 | 12.5 | 29736985 | 13431717086 | COMPLETED |  |
| 2958 | r9 | pointer-ladder | d78b80cb | kg_bench_ladder | 24 | 96 | 25000 | true | true | 0.02 | 128 | 2 |  |  | 860.6 | 1,051 | 65.7 | 12.3 | 29736985 | 13431721173 | COMPLETED |  |
| 2930 | r7 | b48-g2 | 98797536 | kg_bench_ladder | 32 | 128 | 25000 | true | true | 0.02 | 48 | 2 |  |  | 1,105.7 | 1,213 | 95.9 | 9.9 | 29736977 | 39275483188 | COMPLETED |  |
| 2875 | r4 | io-ratio-0.02 | f5110338 | kg_bench_ladder | 48 | 192 | 25000 | true |  | 0.02 |  |  |  |  | 1,422.7 | 1,500 | 192.0 | 7.6 | 29736977 | 39275495361 | COMPLETED |  |
| 2859 | r3 | single-pass | f93fb98c | kg_bench_ladder | 48 | 192 | 25000 | true |  |  |  |  |  |  | 1,489.0 | 1,568 | 166.8 | 9.4 | 29736977 | 39275494691 | COMPLETED |  |
| 2858 | r2 | cached-constants | f93fb98c | kg_bench_ladder | 48 | 192 | 25000 |  |  |  |  |  |  |  | 3,065.9 | 3,132 | 132.3 | 7.4 | 29736977 | 39275580159 | COMPLETED |  |
| 2857 | r1 | inproc-small | c248fed8 | kg_bench_ladder | 48 | 192 | 25000 |  |  |  |  |  |  |  | 4,507.7 | 4,573 | 139.0 | 8.4 | 29736977 | 39275627013 | COMPLETED |  |
| 2856 | r0 | baseline | c248fed8 | kg_bench_ladder | 48 | 192 |  |  |  |  |  |  |  |  | 5,495.9 | 5,565 | 142.6 | 8.1 | 29736977 | 39275658036 | COMPLETED |  |
| 2851 | r0 | baseline | dad37766 | kg_bench_ladder | 48 | 192 |  |  |  |  |  |  |  |  | 11,643.6 | 11,739 | 121.0 | 2.8 | 29736977 | 39275866268 | COMPLETED |  |
| 2876 | r4 | chunk-4e5-single-pass | f5110338 | kg_bench_ladder | 48 | 192 | 25000 | true |  |  |  |  |  |  |  | 265 |  |  |  |  | FAILED | adapters.chunk_size=4e5 |
| 3067 | r10 | order-g8-slim | 4f399ff4 | kg_bench_ladder | 24 | 96 | 25000 | true | true | 0.02 | 128 | 8 | true |  |  | 149 | 96.0 | 11.8 | 1700890 | 2111226149 | FAILED |  |
| 2959 | r9 | full-51-pointers | e783fa14 | kg_uncapped | 64 | 256 | 25000 | true | true | 0.02 | 128 |  |  |  | 15,860.4 | 16,178 | 211.6 | 15.8 | 461629588 | 215337343500 | COMPLETED |  |
| 2889 | r5 | full-51-32cpu | 42e718e2 | kg_uncapped | 32 | 128 | 25000 | true | true |  |  |  |  |  |  | 19,089 | 128.0 | 14.2 |  |  | FAILED |  |
