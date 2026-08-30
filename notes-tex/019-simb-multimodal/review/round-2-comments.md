# Review comments on 019-simb-multimodal_2026-08-27-17-14-39_507ee4b6.pdf

136 annotations, 127 carrying a written comment. Colours are the reviewer's own; the key is the stable handle to cite when recording what was done about each.

### [1] p1 `DG2Z43TN`

> Abstract

It needs to be written to be more concise, more clear. I'm not sure it has taken into account the entire document

### [2] p1 `CZHCDBF8`

> The expression objective is fighting itself: the quantile loss reaches its minimum at epoch 463 and rises for the next 9,500 while Pearson climbs, squared error bottoms alongside the Pearson peak instead, and the model never gets normalized squared error below “predict each gene’s mean” because its predictions are twice as spread out as its own correlation justifies, which a free post-hoc rescale fixes.

So the objective is fighting itself because what MSC and Pearson are at odds? Unclear.

### [3] p3 `QVFJABL8`

> Metabolome head The head that predicts the 19 Mulleder amino-acid concentrations. It appears in two roles. As  the only active head it is the amino-acid strand (section 5). Added alongside the betaxanthin head it is an auxiliary head: it is trained and scored, but the number being reported is still betaxanthin, and the question is whether sharing an encoder with it help

Make it clear if these are just linear probes.

### [4] p3 `VLFQU5WH`

> prosecuted

Is that the right word?

### [5] p3 `RWHXK9ZT`

> The companion pearson_per_instance correlates across features within one strain and stays high under exactly that collapse, so it is never the objective here.

Pearson for instance tells us if we can get strain trends right, doesn't it?

### [6] p3 `VTGLESU2`

> The betaxanthin row is known to be one such case: table 12 carries 0.4301 from a study in a queued project.

Confusing, I thought that this 0.43 was the best beta-xanthanin value.

### [7] p4 `3XL5BYU5`

> No peak has been observed at any budget up to 10,000 epochs.

Yet the mean square error is dropping only at the beginning. It looks like it's minimized at the beginning. Some of them look like they're dropping later as Pierce had climbed, but never back to the minimum that was originally achieved.

### [8] p4 `GIYL8ZSA`

> nmse never returns below 1 after about epoch 400, so the model reaches r = 0.236 while being no better than predicting each gene’s mean in squared error.

### [9] p4 `T7APESIP`

> and a post-hoc rescale by r/s moves nmse from 1.010 to 0.944

explain the post hockey scale more

### [10] p4 `6ILTSBAH`

> 4. The perturbation operator provably cannot express a pair term, and masked unmasking does not  supply one.

This seems like one of the major issues because we're trying to detect changes in gene expression after some neighboring gene within the genome has been deleted. That is, by its very nature, pairwise interaction.

### [11] p4 `5HACDECW`

> every one of those features is exactly zero and the forward pass is identical to the unconditioned model.

This doesn't make sense. Every one of the features is exactly zero. These are measurements, log fold to log two fold change of expression. So how is it that they're zero

### [12] p4 `Z8UF8SBC`

> 201 of them individually above 0.5.

### [13] p4 `3EMFXF56`

> Over the 4,432 deletions the  two screens share, the 19-dimensional pool predicts betaxanthin at out-of-fold r = 0.298, while tyrosine alone, the chromophore precursor, predicts it at 0.064 and correlates marginally at −0.076.

But this doesn't include the predict-only beta-xanthanin arm, which has a near 0.433 correlation.

### [14] p4 `7TT6KRJH`

> Roughly three quarters of the profile’s power survives regressing out single-mutant fitness.

This statement is unclear to me

### [15] p4 `NFJY9IE3`

> On the 724 deletions with no Costanzo fitness record, which carry twice the betaxanthin spread, the profile predicts at 0.455.

We never did any training with fitness in relationship to expression, or not expression, but beta xanthan and production.Why are you bringing up fitness here?

### [16] p4 `3TSD2V2S`

> The metabolome auxiliary head currently costs betaxanthin performance.

Exactly, and this seems to nullify the previous 0.6 point number 6.

### [17] p4 `4G2CXAF9`

> So the coupling exists in the data and the current head does not use it.

Exactly, how to use it, question mark.

### [18] p4 `WUU9M7JL`

> But direction carries real signal: following tflink edges TF → target gives 0.5508 and regulatory interaction 0.5239, against controls near 0.501.

But how to encode direction in a symmetric attention map? Would it have to be made non-symmetric, or would it have to break out multiple maps per direction? Question mark.

### [19] p4 `L6RUGNHY`

> Configuration choice moves results more than method choice. Betaxanthin grid cells span Spearman  0.013 to 0.158 against a between-method gap of 0.0014 on AUC. The amino-acid strand’s best run is 0.211 and its median is 0.0498.

What can this be attributed to sensitive data splits? Also, I think the important thing is that we're using the same method as FCL for data splits. As long as we do that for the comparison then we will be okay and then we can point out the sensitivity to configuration in data splits.

### [20] p5 `KDL9ALDS`

> Figure 6 is ready to be argued on coverage rather than on accuracy, and the amino-acid material belongs in  it as a coupling measurement, not as a joint-training resul

It still seems that training on the joint objective with the amino acid profile could still provide some benefit. I don't think we're done here

### [21] p5 `TEYC2NGS`

> checking whether the graph prior is the right prior at all.

In nine findings you mentioned the graph prior, but is this also covered for expression? We'll have to reread the section

### [22] p5 `LKDA4IE6`

> The graph-prior probe has run and it closed a planned arm. The network-distance pair term is not  worth building, the nine masked heads should be freed, and the mask should stop symmetrizing its two directed relations. Two cheap experiments remain unrun: morphology at full scale, and one expression run trained until the validation curve turns.

The reason I'm reluctant to free the arms, remove the masked heads, as you're suggesting here, is that these networks have been identified in literature in the past and we don't know which phenotypes will benefit from the knowledge brought to bear by these graphs. For instance, protein-protein interaction networks might be beneficial for something, whereas the transcription factor networks or regulatory networks might be beneficial for others. We can also have heads that are not constrained to looking at particular parts of the network. This I think is the only way to guarantee at least that the heads some of the heads are seeing network interactions that are known to be real cell interaction if we have other ideas on this, we can discuss.

### [23] p6 `HPK65P86`

> The sharper finding: the model never beats the mean in squared error. nmse is normalized so that 1 is exactly “predict each gene’s training mean”. It dips barely below 1 at epoch 213 and is back above 1 from about epoch 400 onward, reading 1.010 on v9 and 1.040 on v8 at their Pearson peaks. So the model reaches r = 0.236 while being no better than the mean predictor on a squared-error scoreboard.

Earlier in this section, maybe in the nine findings. You did a correction and the NMSC drops below one. Yep, point to the expression objective is fighting itself. You say post hocke rescale R over us moves the NMSE from 1.010zero to to point nine four four four four

### [24] p7 `W6CBTBUZ`

> a Quantile loss turns at ~500

Fig 1a, it is unclear what grey and yellow are.

### [25] p7 `SUHY7UEP`

> s=r(nmse-optima

S equals R NMSE optimal should just be parallel to the line.

### [26] p8 `XMEVWTTE`

> Consequence, and it is free. Multiplying predictions by r/s changes no correlation at all and moves nmse from 1.010 to 1 − r2 = 0.944 on v9, and from 1.040 to 0.949 on v8. The ordering the model produces is already worth more than its magnitudes, and any report that quotes Pearson beside an nmse above 1 is quoting a model that would lose to the mean until it is rescaled.

### [27] p8 `FKRLU2AA`

> and no term depends on the pair (p, i) of deleted gene and reporter gene.

We obviously need this.

### [28] p8 `769PTZS7`

> Table 5: Pair-rank ladder. The reference arm cannot express any dependence on which reporter is being predicted given which gene was deleted.

Was this fully tested?

### [29] p8 `PBBAPBK8`

> Capacity is not what binds. A rank-r reconstruction of the target matrix reaches 0.7265 at r = 32 and 0.7799 at r = 64, against a best model score of 0.238, so a null on this axis is a statement about the mechanism.

Can you provide more detail here?

### [30] p8 `KY78HZ9I`

> At k = 0 the revealed set is empty, every encoded feature is exactly zero, and the forward pass is identical to the unconditioned model. Scoring happens at k = 0. So the added term contributes nothing at the point the number is taken, and it cannot create a dependence on the pair (deleted gene, reporter gene) no matter how well the unmasking is trained.

We might want to look at scoring at k.

### [31] p8 `EYMIWKAQ`

> and table 6 shows that channel is worth a great deal and is measured to be orthogonal to genotype.

### [32] p9 `CBXWR2VJ`

> A ridge oracle

briefly explain here how the oracle works.

### [33] p9 `G6C3VS7K`

> 0.4562

This is already very good for within chemering only 10 datasets Data points rather

### [34] p9 `9PATB5Y8`

> Figure 2: Residual gene-gene structure in the expression compendium after removing each gene’s mean. The pattern replicates on held-out strains and lives in about 33 effective components, which is what a per-gene independent readout discards.

We need more descriptive figure captions. Also, when labeling figures, we want the letters to be in the top left. Bold, Arial, I think 10 it is. I forget what we're using now. We have a new standard for this. Correct these figures accordingly. It's okay for now if they have some title also. The main reason I'm leaving this comment though is I don't see the purpose of B Just to show the noise across four different genes, unsure. Also, C, we need a better description on this. Also, D, yeah. without the caption it's difficult. E seems pretty consequential, but I can't fully understand it without caption. The result is not conditional on which gene was deleted. It just goes right into the pattern replicates on held out strands., what unclear which is what pur gene independent readout discards Yeah, exactly, okay. this is what we were trying to solve, but the figure caption is still not providing enough. Let me go over F The shuffled null is the floor is F what is being shown in E shuffled null. Maybe F can actually be showing three different figures like is shown here But scratch that unsure what would make this more clear.

### [35] p9 `S8XB7PJW`

> What makes the oracle worth keeping is the structure it exploits. After removing each gene’s mean, the residual genegene correlation pattern replicates on held-out strains at split-half r = 0.8687 against a permutation null of 8.45 × 10−5, and it has an effective rank of 32.78, with 59.1 % of variance inside the top 32 components. A per-gene independent readout emits 6,127 marginals and discards all of it. src: experiments/019-simb-multimodal/results/residual_covariance_  diagnostic.json

But worth keeping how?

### [36] p10 `FVWC79HY`

> residual covariance above measures

Where is the residual covariance measured? I don't see it anywhere.

### [37] p10 `UBIMQ6DY`

> nd the one large available signal is imputation from other genes of the same strain, measured to be orthogonal to the genotype-only task.

Unclear to me

### [38] p10 `EUGL2SIA`

> Cannot say. Whether any of the pair-rank mechanisms help. Whether the graph prior is the right prior for this task. Whether a different objective closes the loss-metric gap.

Should be referencing maybe here the graph prior section

### [39] p11 `ZTBMZNBH`

> morphology and fitness 4,220

Yes, I think we reframe this question to does the addition of training on fitness bump any of the performance in any of the morphology vector.

### [40] p12 `LNBV4S69`

> Pairing it with expression caps both arms at 1,440.

This is only true if we don't mask outputs that we don't have, so we can still backprop on morphology when we don't have expression, if we want. This is a way around this. This has been documented in the past.

### [41] p12 `9DV8XK6F`

> This does not retire the expression-helps-morphology question. That question is what delta_joint_expr_ morph_000.yaml was built to answer, its control is right, and its three conditions (expr, morph, expr_morph) are the correct decomposition. It has simply never been run past 87 epochs. What the census changes is the order: the fitness pairing is where the statistical power is, and it can be run at full scale first.

My original idea for this was to do the masking and have training overall morphology and then to see if adding in the subset of predictions during training on expression could help. This would be a very long run I think and difficult to tune and that's why we just want to nail down expression first Then morphology.

### [42] p12 `MDD7F48W`

> moved by deletions.

Whereas moved by deletions represented in the table eleven.

### [43] p12 `CYMCCMJP`

> Table 11: Top of the scalar-target shortlist. The list is dominated by bounded ratios (actin localization class, bud-size class, nuclear-stage class), which is what a deletion actually moves. Size features are the opposite trade: C115_A whole-cell axis ratio has a ceiling of 0.972 and almost no spread across mutants.

Can you expand this table some and also add the CalMorph description?

### [44] p12 `RMTEJGFD`

> Recommendation: run the scalar warm-up on A113 or C124_C, both of which are reliable and genuinely variable, and treat the run as a convergence-budget calibration for the vector target rather than as a result.

Yes, I think this is a good idea. let's give this section a green check after making changes to my comments based on my comments

### [45] p12 `C4AL6YLC`

> reliability

I don't get the difference between reliability and the Pearson ceiling. Yeah, maybe define somewhere early on how this is defined

### [46] p13 `HPKMTKA8`

> which is 0.68σ against a measured replicate σ of 0.0302, so it is a plateau rather than one lucky trial

I don't follow this logic. It needs to be clear. 0.68 sigma is Its top five trials span this 0.02 range, which is 0.68 sigma against a measured replicate of 0.03. So it is a plateau rather than a lucky trial. You're assuming that the top five trials having or what is meant by trial here splits of the data yeah we need a better table caption.

### [47] p13 `XXEIM3PQ`

> That costs 537 training strains,

Why don't we get these other strains back into training? I mean there were valve test strains before, right? So what's the deal? Why is the end train smaller? on the old model.on v4

### [48] p13 `PIZBQIXS`

> Using the 0.4301 model instead would mean scoring on genes it had trained on, which is the one thing that would void the comparison.

Oh, I think we just need to do a different shuffle. So we can use their data set for test and their split for train plus all the other genes that we could use for train. And we can add some additional genes to keep the splits proper, maybe for 0.8, 0.2, or whatever the splits are, and add them also into test. And then for the evaluation, we can just take the subset of their genes and then also do just the full regular 0.8, 0.2 eval.

### [49] p13 `HAQRQYFW`

> Both numbers are real and they answer different questions: 0.4301 is what the model does on this screen, 0.372 is what it does with a competitor’s test set held out.

But only on training on a data set that's smaller than necessary, right?

### [50] p13 `S5U68X8J`

> Both curves approach the base rate by k = 150.

Is this what is being shown in figure 3 with the purple line

### [51] p13 `59YB7KNL`

> Their rank correlation is −0.128, which at n = 639 is z = −3.2, where two models correlating with the truth at +0.105 and +0.039 would through the truth alone correlate with each other at about +0.004.

I don't get the main point being made here when it says with truth at this, what does it mean with the truth at? Correlating with the truth at this and this, with through truth alone correlate. It's very confusing.

### [52] p13 `7EVZSC46`

> No mechanism is claimed. for the sign. What is claimed is the operational consequence: recall is near-additive because the two methods are finding largely different genes, which is an argument for running both, and which is invisible in every aggregate statistic. The 51 that neither finds is the honest ceiling on this comparison.

Clarify the right, I'm confusing here

### [53] p14 `MYIHVZNI`

> The nine unselected CGT grid cells span 0.10 to 0.65 at k = 50, so cell choice matters more than the gap between methods.

Some of the yellow lines look concerningly below purple very early on. I see one of them, maybe two of these lines that are kind of overlapping. What is going on with those? They cross over under 25, so it's that they have a barely better than random precision at K early on. I mean, some of them even look like at the top, they're below. I guess that's maybe to be expected in the top one or two. Clarify.

### [54] p15 `NEWNCH9B`

> identical derived labels

We need more description on the methods of deriving labels earlier on. LaTe, use the exact method, explain the differences, even potentially a graph for this.

### [55] p15 `4S8WG8B8`

> r outside the model than inside (+0.270 against +0.124),

Is this outside of FCL, meaning outside of their training set? This is confusing what outside and inside the model mean.

### [56] p15 `CREPJEFM`

> Fisher r-to-z gives z = 2.07, p = 0.039.

Don't get this part

### [57] p15 `3GE55GUY`

> And labels for the non-GEM genes are derived by us, since the FCL paper never labeled them, so the comparison to make is panel against panel.

Yes, FCL doesn't label them, but we can use the same kind of labeling scheme.

### [58] p15 `2BJFHRVU`

> The selection is asymmetric, and it favors FCL.

Whole section needs to rewrite unclear, poorly written.

### [59] p15 `I7FJIKCF`

> and their validation folds appear to be drawn from the same 640 genes as their test list

Their test and val are the same. You need to double-check this. We want to have the same setup as them, so we plan to do this again in a more well-structured way, training all necessary models to be able to do the comparison. Some of the things I think we missed before, which is that we should make sure the splits are set up properly. I think we had to do this kind of in a post hoc way, maybe, where the total number of training data ended up getting reduced because it was actually in the test set or something like that, unsure.

### [60] p15 `8GY22JGM`

> Where the two land level, we are level with an opponent picked with more information than we allowed ourselves.

Unclear what is meant by this

### [61] p16 `6XG6KXBY`

> Validation Pearson does not track test AUC. Four of ten CGT cells beat FCL’s AUC of 0.570 and the best  reaches 0.695, while the validation-selected cell reads 0.557. The runner-up on validation, 0.3528 against 0.3639, has an AUC 0.14 higher. Neither “CGT’s AUC is 0.695” nor “CGT’s AUC is 0.557” is safe; the metric does not separate the two methods at this resolution.

Is this another way of saying the data splits matter? I think earlier you called this conflict. Make it clear. I think this section isn't also written very well.

### [62] p16 `VGTT22XL`

> • Their labels are a headroom bound. Their binning rule min-max scales production and cuts at 0.40 and  0.65, so applied to our larger copy of the same screen it gives 107/476/56 against their 109/431/100. 18.8 % of genes sit in a different bin than our values would give, so a model predicting our measurement perfectly is marked wrong on about a fifth of genes.

What to do about this, we want to be able to make this comparison fairly. Maybe we don't apply the scheme exactly, we just apply the value at which they are cut instead of the 0.4, 0.65 min-max scale, we just take the value, the transition between low, medium, highWait. Don't we correct this? It says below, regression versus classification. We use rank based or split after

### [63] p16 `RUNXG5VV`

> The comparison is therefore between a regressor and a three-class classifier, which is why every metric above is either rank-based or computed after binning our predictions with thresholds fit on the training split only.

We do need Some explanation over the difference between rake bait rank based and also the Binning based trending split only. Is this why we get this mislabeling problem on the previous point?

### [64] p16 `P4J8U482`

> Established. On this screen CGT and FCL are tied where both can be measured, both have tail-confined signal at roughly 5× enrichment in the top 25, and they find different genes. CGT predicts the 81 % of the screen a flux-based method cannot represent, and finds high producers there at 4.4× over base rate.

I like the gray boxes that we had been using in the microbe perturb seek. I think let's put major points at ends of critical sections in these grey boxes, like this here

### [65] p16 `HQDCA39E`

> That the absolute top-100 membership is a deployable call, since the rank-matched comparison hands both sides the true class marginal.

don't understand this last point, especially the last clause. The rank matched comparison, hands both sides the true class marginal.

### [66] p17 `RJK48J4X`

> These runs converge, which the expression runs never do 7

I think it's hard to say that they converge because we thought that the expression runs converged and the MSC does dip early and looks like it converges. Even Claude was saying that they're converged and there's only after running for thousands of epics do we realize that the piercing can climb and that the NMSC can eventually start to drop again, also.

### [67] p17 `HUWX8SPB`

> The mechanistic prediction is that a deletion’s free amino-acid pool carries information about how much betaxanthin that same deletion makes.

But the fact that the beta xanthan production never gets all the way up to predicting on beta xanthan alone doesn't necessarily indicate that the whole pool is helping. It could just be that we can learn on the beta xanthan signal and in fact, the other signals aren't actually helping, but they're hurting. So I don't think we can make this statement, right?

### [68] p17 `SDPV8HSE`

> Repeat over three fold shuffles and report the mean.

You should be reporting some plus minor standard deviation also, right? Please put this in the table too

### [69] p17 `G8U7UL3H`

> So r = 0.298 means: knowing a deletion’s 19 amino-acid concentrations lets you predict its betaxanthin at r = 0.298 on deletions you did not fit on.

### [70] p18 `3U75MZID`

> A slow deletion can have a distorted amino-acid pool and a distorted pigment readout with no metabolic relationship between the two, which would produce exactly the correlation above for an uninteresting reason.

Is this saying something like fitness is a hidden cause? Unclear.

### [71] p18 `WV796MDA`

> Regressing it out of both sides leaves the amino-acid profile at 0.133 against 0.176 without the control on the same genes, so roughly three quarters of its predictive power survives.

I don't know what regressing it out of both means.

### [72] p18 `6J8VNSTR`

> The control also shrinks the gene set, from 4,432 to the 3,708 deletions that carry a Costanzo record, which is why both the controlled and uncontrolled fits are reported on that same 3,708 rather than compared across different populations.

We also have Kuzmin SMF if this has more overlaps or can fill in some of the gaps since it's the same method.

### [73] p18 `N8XZ3G7Q`

> against 0.464 for the rest

This means the ones with overlap in the beta xanthana data set. Question mark.

### [74] p18 `WDPRZLAH`

> This is the population that carries the extremes, and it is the population a fitness-based control cannot reach.

We can check our other single mutant fitness profile data sets.

### [75] p18 `XUR99U6G`

> Figure 7: (a) Marginal correlation of each amino acid with betaxanthin over the 4,432 shared deletions; tyrosine highlighted. These reproduce experiments/019-simb-multimodal/results/pigment_noise_ceiling.json exactly, which the generating  script asserts rather than assumes.

We need more figure description here. This uses the ridge regressor from 5.3, right?Also, we want the correction of having the labels in the top left with having basically a white marginal cross around the labels and they need to be properly sized to Arial font, we can keep the title that has the main point for the figure.

### [76] p18 `YMTBGK2W`

> (b) Cross-validated ridge fits from table 15.

Purple and grey are confusing. I think a legend would do good here instead of referring to the table. Is purple supposed to be the regressed out version? 0.133 or 0.133 and then fitness alone is 0.151. Okay.

### [77] p18 `3IHVUXKM`

> (c) Median tyrosine by betaxanthin decile. The relationship is non-monotone: tyrosine is highest in the lowest-production decile, falls through the middle, and rises again at the top, which is why a linear correlation reads near zero. The axis spans 2.65 to 2.89 mM, so the swing is about 9 % of the median.

Better caption. Still need to fix the C label

### [78] p19 `BWPPS6IV`

> Table 16: Betaxanthin score with and without a 19-amino-acid auxiliary head, paired within grid cell, on val/betaxanthin/pearson_per_feature.

I don't much get the point of this table when it's basically just filled with NANDs. And point three, the auxiliary head is weak. Primary is good. Sure, it's not that point, it's a different point. Yeah, I don't know why we have this table. It's I don't know if it's really providing much of anything. I think we haven't properly explored the relationship between the decoder. We just did something extremely naive, so we don't need to overanalyze this part. We just need to Be more concise and clear about possibly where to go next given the structure of the data.with a brief mention over what was tried. Just this metabolome head, see what happens. Doesn't look like we outperform with Kachera. Beta xanthin prediction probably actually hurts.

### [79] p19 `KJKAJCUB`

> regressing out single-mutant fitness.

still need to know what regressing out means. It's fine if you just communicate it with me or state it clearly within the paper, but I get the idea it's somehow removing the effect or the predictive power of fitness. Need more detail.

### [80] p19 `AGIPNSZB`

> Whether a shared encoder can exploit the coupling: the data says the coupling is there, the joint training says the current head does not use it, and at n = 1 per cell those two are not yet in contradiction.

Couldn't it be that the correlation that we're seeing from the amino acid pool is completely covered by the Cachera data set, so that we could never do better than Than training on the Kachera data. We have no notion here that it's the amino acid pool is actually synergistic, right? We get this near point three correlation related to beta xanthin production, but all of that might be covered by the Kachero correlation. That was not made clear in this section.

### [81] p20 `7LJLMHD9`

> Thorndike

What is this, a different paper?

### [82] p20 `ZYVXISZ7`

> Read the 0.075 as what a re-screen of selected winners looks like under range restriction, not as evidence that the score is worthless.

Need more explanation on this

### [83] p20 `65IINSMJ`

> Figure 8: Reliability of both pigment targets. The betaxanthin panel uses the per-record standard error over a median of 15 colonies; the beta-carotene panels use rank agreement, because the target is a hand-scored ordinal.

Why does figure 8 have beta xanthin? We're talking about beta-carotene in this figure. This is in the 6.1 beta-carotene section. No explanation, figure caption is obviously bad like the others.

### [84] p20 `CVPADY7A`

> t 0.118 against a chance level of zero, and its best run peaks late (epoch 791 of 999), so it shares the expression strand’s under-training rather than the metabolite strands’ early overfitting.

Where is this coming from?Not convinced by this argument

### [85] p20 `JE46MC9E`

> What the strands have in common

I think we can shrink this and simplify this section sum. I think figure 9 provides some value. I don't get really figure ten expression mask has a peak epic as a function of the run. Oh, peak epic. Subtraction of the wrong. How many epics it actually takes to change these things? Okay. Thousand? Well, figure ten is fine.Specifically table 19, maybe there's a better format to put this in than just a list in a table.

### [86] p23 `NEE4PJ9I`

> The 010 fitness and interaction task shows neither behavior, with both losses falling monotonically and the validation-loss minimum at the last epoch.

You have to make sure you actually check this. I'm not sure this is true

### [87] p23 `YLZXLINU`

> under-shrinkage rather than incapacity

I need to understand these terms better. Can you, like in microbe seek perturb seek PDF, put at the beginning a table of contents? There's other terms too, like regress out and things like this.I also think that maybe marginal is being used in multiple ways. Marginal probability as in the sum over probabilities collapsing one of the dimensions versus marginal as in small. Confusing sometimes.

### [88] p23 `N6V9E7TY`

> the last column is where nmse lands after multiplying predictions by r/s, a post-hoc rescale that changes no correlation at all. 1 − r2 is where nmse would land after multiplying predictions by r/s, a post-hoc rescale that changes no correlation at all, and is the optimum for a predictor that is a purely scaled version of a correlated signa

This looks fine to me, but I need some explanation of the logic somewhere.

### [89] p23 `ZDCIRV2S`

> expression at 0.24 sits at 1.01

But if you do the rescaling, then you do beat the mean, so how to properly think about this.

### [90] p24 `E5J9TIDL`

> The lower the achievable correlation, the further predictions drift above the spread that correlation justifies, which is what under-shrinkage looks like when most of the target is noise.

We can need proper definitions of under shrinkage.and related terms.

### [91] p24 `4WX4CS64`

> On betaxanthin the ten grid cells span Spearman 0.013 to 0.158 and AUC 0.406 to 0.695, against a between-method gap of 0.0014 on AUC.

Again, are grid cells trained test splits?How do they differ?

### [92] p24 `L8TXEC8M`

> cell is quoted,

Which split?

### [93] p24 `5RJGGBZC`

> Table 22: Re-running an identical configuration moves the objective by σ ≈ 0.030. The observed spread across configurations is several times that, so configuration choice is doing real work; but any single difference under ≈ 0.03 is inside this floor.

I'm getting confused now. Configuration is model configuration, correct?

### [94] p24 `ZI8YMZ92`

> Genuine configuration effects:

Okay, now it seems that you're using grid as hyperparameter config grid.

### [95] p24 `783YQUQX`

> selection inflation

We'll need definition in terms part.

### [96] p24 `EFQKM9NM`

> Before spending compute on joint proteome and expression training, the linear overlap between the two was measured on the 1,350 strains they share. Per-gene correlation across strains has median r = 0.08, with 1.7 % of genes above 0.3 and 8.9 % negative. A ridge map recovers held-out R2 of 0.035 in one direction and 0.024 in the other, above a trivial baseline of −0.004 but far from redundancy.

Which expression data is this? Unclear. I wasn't yet thinking of doing joint training over expression data and proteome data. But it's something I've considered for far future.

### [97] p24 `NAHSM3CR`

> The two modalities were measured in different media, synthetic minimal for the proteome and synthetic complete for the expression, which attenuates every number above. The overlap is therefore certified as non-redundant and not certified as cleanly complementary. That distinction is what bounds how hard a “the proteome helps expression” claim can be pushed until a same-media pair exists.

Yeah, provide proper citation here

### [98] p24 `QK49AYPM`

> 7.7 The graph prior has now been checked, and it is at chance 7

Instead of presenting it in this form as has now been checked, I think that we need to integrate to the entire document to just refer to this as being later investigated, or we investigated this because of this, and this is what we need.

### [99] p25 `N2XAPWD9`

> Figure 11:

I'd like to try to do our best to force the figures that belong to a particular section to be within that section, otherwise it creates some confusion

### [100] p25 `S9H9NSSQ`

> Figure 12:

Like all of the figures, we want these to be properly labeled ABC which has a white cross of margin with area font adhering to nature biotype publication standard labels' top left. This helps the plot itself be easily cropped out. Titles can remain.

What is the target here? I think we should probably be doing this for the other genotypes as well to make the arguments more clear. So the others from seven morphology beta carotene amino acid. We're doing expression here. Beta Santhan. All the ones mentioned from the group. That way we can see if the graphs have effect on some but not other phenotypes.To do this we can expand this into a six panel.

### [101] p25 `IDYRGAWG`

> a) All nine graphs in the orientation the model actually uses.

The bars look like they are overlapping each other. They need to be flush.

### [102] p25 `58T2FWCB`

> so it makes no prediction for them

This is the point of also having open attention maps.

### [103] p26 `B4HF3N8L`

> Nine attention heads are currently constrained toward a target that does not predict response.

strictly for the expression case, right? We want to look at this for the other cases.

### [104] p26 `Y2LRBRYS`

> stop symmetrizing

Since the attention map is symmetric, I guess we will need some strategy for this. I guess it's normally symmetric, but it doesn't necessarily have to be.

### [105] p26 `PESXUDXG`

> and the kNN probe already showed the perturbed side failing to use structure that was present.

I think we should probably be including the KNN probes result into this document

### [106] p26 `2AXJ26NI`

> The strands that work in this project are the ones where the supervision constrains a gene repeatedly.

### [107] p26 `XXEFLKFD`

> That predicts the gain should be larger where the two phenotypes share strains and smaller where they merely share genes, which is a testable statement and has not been tested.

meaning we have expressional morphology for one knockout strain versus What would it what I don't get it? What does it mean when two phenotypes merely share genes? S in reporters? doesn't make sense.

### [108] p26 `J6V2MS45`

> The first four are cheap enough that not running them is the expensive choice.

### [109] p26 `5BBPQSH5`

> The graph-prior probe is DONE, and it came back at chance

We will need these with respect to other levels as well to make any statement about this. The reason is these are graphs that are constructed from historical data and other inference methods, and if they are useful for some, we don't want to just remove them for building up to the multitask training case.I think we would rather just have them if they provide marginal signal good and then we can have alternative attention self-attention that is unrestrained by masked graphs

### [110] p26 `M3SW8R6S`

> The one repair with evidence is to stop symmetrizing the two directed relations: TF → target reaches 0.5508 on tflink and 0.5239 on regulatory interaction, and _build_attention_mask currently averages that away against its uninformative reverse. That is a change to the mask builder, not to the model.

I agree, let's have some fix for directional graphs.

### [111] p26 `I3AZKZ6B`

> Morphology at full scale. Drop require_modalities for the morphology-only arm and train on all 4,718  Ohya deletions instead of the 1,440 that also carry expression. This is a config change, and it is the only way the 0.0824 becomes a number about morphology rather than about a quarter of morphology. Run it long enough to see a peak, which section 7 says cannot be assumed from the expression budget.

This training run will take long and I want to focus on expression first So we are going to table it see if we can learn as much as we can from expression.And also, I want to do some of the I want to do the beta xanthanin and the metabolite pool, the amino acid pool, and see if there's any synergism by adding additional labels there. The reason for this is these runs are also smaller for training, faster to train, faster to get results back, less complicated with fewer labels

### [112] p26 `88CZBF3E`

> colony size is fitness

What I meant before was just a cell size.But I think that this term actually has a little variance across the entire data set. I like this idea, but I think we should expand. I want to see a larger table, maybe top ten, top fifteen, top twenty of the different features with descriptions in this document.

### [113] p27 `R6PNIPQN`

> Apply the post-hoc rescale to the existing best checkpoint. Multiplying predictions by r/s changes no  correlation and moves nmse from 1.010 to 0.944, taking the model from worse-than-the-mean to better-than-themean in squared error (section 2.3). An evening’s work, no training, and it removes an objection the figure would otherwise have to answer.

We don't consider this the final figure, the final training for this expression. We're trying to use this document to know what to do next with expression. We will do this eventually if need be

### [114] p27 `PDCBMYCY`

> 5. Bracket the expression peak, by resuming rather than restarting. No run has observed a maximum  at any budget up to 10,000 epochs, and a fresh 10,000-epoch run costs 3.8 days. Resuming the existing best checkpoint is the cheap form of the same experiment, and the epoch at which the curve turns is what sets every arm budget afterward. section 9 proposes E = 4,000 in the meantime, justified by an arm that has been observed to peak at 3,921.

Yes, I think we can actually do this now, as in today, August 30th. My fear is that the Pearson is improving so slowly, but let's just do it anyways. The other thing that we could potentially do is we can take the checkpointed model and we can train it with DDP over multiple GPUs to go faster. We will have to check the available GPUs on the MMLI partition on IGP cluster IGB

### [115] p27 `VMGLGAGH`

> • Then the pair term (table 26). The reference operator provably cannot express a dependence on the pair  (deleted gene, reporter gene), and masked unmasking does not supply one at the point the score is taken. A rank-64 pair term already reaches 0.2274 in 2.0 days against the masked objective’s 0.2362 in 3.8.

I'm still a little bit about fuzzy on how attention cannot solve this. We want the pairwise operation or some pairwise operation between the gene that was deleted, and then all of the genes in the genome. You're telling me this only works if we have greater than one gene?

### [116] p27 `FQD7P5RW`

> • Pair morphology with fitness for the first full-scale joint test, since fitness shares 4,220 strains with  morphology against expression’s 1,440. Then run the expression pairing on the 1,440 as the mechanism test it was designed to be.

I think this is good, and the question we should be asking ourselves is, does the addition of fitness improve the prediction performance of morphology at all? Any of the values of morphology, or basically on average, does it improve morphological measurement?

### [117] p27 `JNQZF9RW`

> Keep it as a negative control, not as the headline contrast.

Yeah, we do have to keep it as negative control.I think things will probably change some too when we add in the metabolic representation since these things are linked up by pathway.

### [118] p28 `RCH5WVW9`

> The masked-conditioning oracle reaches cross-study 0.4838 at m = 1000

I am not too impressed by needing to know a thousand genes to predict the expression of the next couple thousand. Collecting the expression values for a thousand genes is already pretty expensive.

### [119] p28 `4MX2DVSZ`

> defensible product surface

What is meant by this defensible product surface?

### [120] p28 `EI3HV6Z6`

> The manuscript currently claims r = 0.543 for expression and 0.619 for morphology inside a paragraph marked [FILLER – R3.]. The measured values are 0.238 and 0.082. Neither placeholder should survive into a draft anyone reads.

Yes, we need to remove the placeholders from the manuscript

### [121] p28 `3HK24LYH`

> A 0.24 per-gene Pearson is not a Figure 3 headline.

I agree with this, that's why we want to continue to work on it.I think we need to bring these other pieces of evidence to bear on what to do next. So this is the data that you just provided, that if you know 100 or 1,000 genes, this masculine will imputation. You can get decent correlations, but then also the k-nearest neighbors result, and we should use this in helping to guide some of our architectural designs.I am hesitant to get rid of the graph information that we include. Also, previously I was using in 010 the graphs as a label to be predicted on top of the attention map. We decided that masking the attention is basically the equivalent of this, but not having to actually learn the data. I'm wondering if displays any sort of structural difference in the representations of the network.

### [122] p28 `6A6MVH2F`

> Two of the three cheapest routes to a better one are not modeling work at all. Morphology has never seen more than 1,161 of its 4,718 strains in any of 397 runs, so the full-scale run is a query change and is the largest single change available to the figure.

We don't want to do morphology yet, just because it's going to be another long training run. We'd rather focus first on expression. Morphology has nearly 4x the number of strains and it's still predicting vector with similar architecture size. This will take much longer. And the expression fit at some 10,000 epochs. So we want to focus on expression first. We've already made the most headway with this and I think we are triangulating in on the issues.

### [123] p28 `G62MEQRE`

> And the graph-prior probe decides whether nine attention heads are enforcing a premise nobody has checked.

This has already been done, but it needs to be elaborated because I think it was only done with respect to expression

### [124] p28 `H2TLULLK`

> The argument is coverage: yeast-GEM reaches 19 % of the screen and misses 73 % of its high producers, and CGT finds high producers in the part a flux method cannot represent at 4.4× over base rate.

### [125] p28 `TELTSP8K`

> That last one is the constraint-based layer above: the amino-acid coupling is real in the data and unused by a plain auxiliary head, and a stoichiometric constraint over yeast-GEM is what would close it

I think adding this would be pretty powerful too because then it basically covers all of our use cases. We'll be able to do metabolism, we will be able to do predictions with respect to gene states like expression and protein self-fitness at the global level with the class token as in fitness value or interactions from subset of gene representations or even chemogenetic interaction as an interaction between the chemical and gene, etc. As soon as we have metabolism, everything can basically run through it, so it's a matter of convenience, not necessarily a matter of selecting the right model.

### [126] p28 `AZMB3I6T`

> Keep the option text in the draw.io boxes. Several of those panels name alternatives that have not been run, and a composition that still reads when one of them does not arrive is worth more than a tidier one that does not.

I think the panel we need is what happens when you add fitness to morphology.Does it bump the mean morphology performance at all?

### [127] p29 `UE3UXKBP`

> What E = 4,000 gives up, stated plainly. The v9 masked run was still climbing at 9,674 and reached 0.2362; truncating it at 4,000 would have scored it near 0.21. So a 4,000-epoch comparison is fair between arms and understates the absolute number by roughly 0.02 to 0.03. That is the right trade for an arm comparison and the wrong one for a headline figure, which is why the winning arm gets one long run afterward.

Instead of doing this, the first thing I think that we should do is probably just load this model from checkpoint and continue training. Let's try first on one of the CABI GPUs on Delta or not Delta, sorry, on IGB.

### [128] p29 `92ZCIRE9`

> O_corr correlation objective optimizing the quantity actually scored

But are there any good ways of actually doing this? And it seems like it's not actually often done because it's used as a metric outside of the objective.I literally never see people optimizing correlation in the major publications that I've read

### [129] p29 `YD8A53QZ`

> Report nmse and the SD ratio on every arm, not just Pearson. The calibration identity in section 2.3 means an arm can raise Pearson while leaving nmse above 1, and that is a different result from one that fixes both. Both numbers are already logged.

### [130] p29 `BJMUV8PD`

> Free, and worth doing on day one. Apply the post-hoc rescale by r/s to the existing best checkpoint. It changes no correlation and moves nmse from 1.010 to 0.944. It is an evening’s work and it removes the “loses to the mean” objection from the figure regardless of what the campaign finds.

We can do this later

### [131] p29 `JLYIL3AX`

> The gate on P_graph has been resolved, and it closes. The probe has run (section 7.7). Graph proximity does not predict which reporters respond, on any of the nine graphs, in the orientation the model uses: AUC 0.4961 to 0.5057 against degree-preserving controls, and longer walks make it worse. Do not build P_graph, which would have made the pair term a function of network distance on a network that does not carry the relationship. Drop it from Phase B, which takes that phase to three arms and six slots.

This isn't settled in my mind. We have to look at the other labels. Labels

### [132] p29 `8T9IUZFN`

> stop symmetrizing the two directed relations

We should definitely try this

### [133] p29 `M38VGUMG`

> Neither costs a new mechanism.

Don't we need some different mechan mechanism to s symmetrize or avoid symmetry in attention?Maybe the mask can just be constructed in an asymmetric way.

### [134] p30 `C5DRMNP6`

> 3. Run the scalar warm-up in parallel, on A113 (actin_n_ratio, ceiling 0.873, robust CV 2.37) or C124_C. It  calibrates the epoch budget for the 278-feature target for a fraction of the cost.

When we get to it, I think this can be done first. But again, we want to put this on hold so we can focus everything on expression. We want to just solve expression first

### [135] p30 `2L3MNPRR`

> 9.7 The joint question, which is the one Figure 3 is about 7  Does expression help morphology. The control already exists and is correct: delta_joint_expr_morph_000.yaml pins all three conditions to the 1,440 genotypes carrying both phenotypes, so expr_morph minus morph isolates the auxiliary head from a data-quantity difference. It has never been run past 87 epochs.  Run it at two scales, and do not conflate them. On the 1,440 shared strains it is the mechanism test it was designed to be. Paired with fitness instead of expression it keeps 4,220 of the 4,718 strains, three times the instance count, which is where the statistical power is. The two answer different questions and both are worth a slot; the fitness pairing is the one likely to produce a detectable effect.  The prior from the metabolite strands is not encouraging, and should be said out loud. The one auxiliary-head experiment run to completion anywhere in this project is the metabolome head on betaxanthin, and it reads −0.0265 ± 0.0159 (table 16). A second phenotype sharing an encoder has not yet been shown to help any first phenotype here. The expression-morphology pairing has a better prior than that one, because the two phenotypes share strains rather than merely genes, but the campaign should be designed to measure the effect rather than to confirm it, which means the control arm matters more than the joint arm.

Again, this is going to be after individual morphology training. So we will have to revisit this in future anyways

### [136] p30 `7NHJ2R8A`

> No objective and no pair term changes that. If Phase A and Phase B both come back inside the noise floor, the honest reading is that the binding constraint is the data rather than the model, and the figure’s expression panel should be built around the ceiling-relative statement and the imputation capability rather than around a genotype-to-expression number.

If we end up restructuring around imputation, we will want to see how far we can drive down the number to a set of the most important genes that will give a representation of the cellular state. But I guess that we can already do this by just looking at the expression matrix, can't we? we could already do this by basically sampling across different sets of genes, different sets of whatever ten genes, say, take one of them, samples, since we probably can't do all combinations. and or we could select genes based off of their ease of measurement or something like this. The fact that they can all be easily detected via mass spec or HPS C Echo Mass spec that is or HPL C .... still for future.. but i think we Still need to gather the k-nearest neighbors and this oracle and flesh out the details some about exactly what was done and I would like a figure constructed for this.

