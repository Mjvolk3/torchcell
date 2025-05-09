---
id: 6tjs280dlb6pm3pf6m7tk1z
title: 020439
desc: ''
updated: 1744961008104
created: 1744959884313
---

## Comparison of SGD Networks with STRING Networks

### Physical Graph (139,463 edges)

| STRING Network | STRING v9.1 Edges | Shared | Jaccard | STRING v12.0 Edges | Shared | Jaccard |
|----------------|-------------------|--------|---------|-------------------|--------|---------|
| Neighborhood   | 45,610            | 2,108  | 0.0115  | 147,874           | 4,578  | 0.0162  |
| Fusion         | 1,361             | 153    | 0.0011  | 11,810            | 788    | 0.0052  |
| Cooccurence    | 2,664             | 476    | 0.0034  | 11,115            | 1,005  | 0.0067  |
| Coexpression   | 314,013           | 21,555 | 0.0499  | 1,002,538         | 69,946 | 0.0652  |
| Experimental   | 219,995           | 59,886 | 0.1999  | 825,101           | 105,034| 0.1222  |
| Database       | 33,486            | 7,935  | 0.0481  | 73,818            | 11,230 | 0.0556  |
| Combined       | 536,207           | 68,743 | 0.1133  | 1,298,235         | 106,455| 0.0800  |

### Regulatory Graph (9,745 edges)

| STRING Network | STRING v9.1 Edges | Shared | Jaccard | STRING v12.0 Edges | Shared | Jaccard |
|----------------|-------------------|--------|---------|-------------------|--------|---------|
| Neighborhood   | 45,610            | 9      | 0.0002  | 147,874           | 27     | 0.0002  |
| Fusion         | 1,361             | 1      | 0.0001  | 11,810            | 11     | 0.0005  |
| Cooccurence    | 2,664             | 1      | 0.0001  | 11,115            | 37     | 0.0018  |
| Coexpression   | 314,013           | 153    | 0.0005  | 1,002,538         | 1,050  | 0.0010  |
| Experimental   | 219,995           | 489    | 0.0021  | 825,101           | 1,244  | 0.0015  |
| Database       | 33,486            | 58     | 0.0013  | 73,818            | 65     | 0.0008  |
| Combined       | 536,207           | 608    | 0.0011  | 1,298,235         | 1,514  | 0.0012  |

## STRING Network Overlap Analysis

### Number of Edges Appearing in Multiple Network Types

| Number of Network Types | STRING v9.1 | STRING v12.0 |
|------------------------|-------------|-------------|
| 1 network              | 467,361     | 630,567     |
| 2 networks             | 58,079      | 572,039     |
| 3 networks             | 9,561       | 85,339      |
| 4 networks             | 1,115       | 9,900       |
| 5 networks             | 79          | 346         |
| 6 networks             | 12          | 44          |
| **Total unique edges** | 536,207     | 1,298,235   |

### Pairwise Network Overlaps

| Network Pair                       | STRING v9.1 | STRING v12.0 |
|------------------------------------|-------------|-------------|
| Coexpression ∩ Database            | 15,409      | 57,388      |
| Coexpression ∩ Experimental        | 34,806      | 557,632     |
| Cooccurence ∩ Coexpression         | 1,251       | 7,119       |
| Cooccurence ∩ Database             | 395         | 1,184       |
| Cooccurence ∩ Experimental         | 874         | 5,660       |
| Experimental ∩ Database            | 11,940      | 40,076      |
| Fusion ∩ Coexpression              | 482         | 8,601       |
| Fusion ∩ Cooccurence               | 77          | 318         |
| Fusion ∩ Database                  | 243         | 876         |
| Fusion ∩ Experimental              | 345         | 5,960       |
| Neighborhood ∩ Coexpression        | 18,694      | 120,681     |
| Neighborhood ∩ Cooccurence         | 255         | 287         |
| Neighborhood ∩ Database            | 3,552       | 22,212      |
| Neighborhood ∩ Experimental        | 5,797       | 61,904      |
| Neighborhood ∩ Fusion              | 302         | 1,678       |
