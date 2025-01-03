#ArsRete_Main
#This is the main function of the ArsRete project
#It imports from other files, but in general everything should be run from this file.
#Additionally this is where I keep notes on the status of runs and the project as a whole.
#About the Project:
#ArsRete is a machine learning program that takes lists of words as input and trains a generator to produce
#outputs that resemble the inputs.
#The intended purposes for producing fictional words are to generate things for gaming and writing, though
#this is by no means limited to that use.

#__RUN NOTES___

'''
__ORIGINAL PROJECT ERA__
1/24-2/28

These notes are based on runs done in an older version of the project before it gained its present structure.
Numbered runs are run from a version of the ParameterSearch() function, which runs a range of neural network architectures
on a fixed random seed and compares the results.

EARLY EXPERIMENTS: These runs did not use a fixed random seed.

__VARYING SIZE OF HIDDEN LAYERS__

Test #1: One generator hidden layer of sizes 100/125/150/175/200. Smaller slightly better, but minimal variance.
Test #2: One discriminator hidden layer of sizes 100/125/150/175/200. Smaller still slightly better, but still minimal variance.
--Adds about 1% to the overall quality metric for each 25 cut off the size.
Test #3: Two-layer generator, 100/125/150/175/200 for the first and 150 for the second. Slightly stronger towards the middle. Does this indicate
--anything special about the 150 case, or does the fact that the generator's 1st and 2nd layers were uniform make a difference?
Test #4: Trying run #3 again with second-layer size of 200. 175-200 outperformed the smaller layers here.
Test #5: Two-layer generator, 100-200 at increment 25 for 1, and 100 or 200 for 2. No conclusive results.

__THEORY 1__: For this application making the generator layers of uniform size is better than not.

__VARYING DEPTH OF NETWORK ELEMENTS__

Test #6: Varying depth, 1, 2, or 3 hidden layers each for generator and discriminator, with sizes set at 100 for speed.
--Shallower generator and deeper discriminator performed slightly better.
Test #7: Repeating #6 with layer size 150. Shallowest generator did perform well, but gen 2/disc 1 also did well.
Test #8: Repeating #6 with layer size 200. Shallowest generator did perform well, but gen 3/disc 2 also did well.
Test #9: Attempted to repeat #6 with layer size 300, no data due to bugs.
Test #10: Run with generator size 1,2,3 and discriminator size 3,4,5. Diminishing returns seemed to emerge, 1,3 and 2,4 were the best performers here.

__VARYING DEPTH WITH RATIO 1:1__

Some GAN architectures I've seen update the generator more per epoch than the discriminator.

Test #11: Gen and disc depth 1,2,3, size 100, ratio 1:1. Best performer was 1,1.
Test #12: Repeated with size 200, 1,1 and 1,2 remained strongest performers.
Test #13: Gen and disc depths 3,4,5 with size 200 and ratio 1:1. No data due to bugs.
Test #14: Depths back to 1-3, layer size 500 and mbd regularizer increased from 8 to 12. 1,1 did the best early, but 3,2 caught up in the later stages.

Depth may be something of a dead end, these never vary by that much.

Test #15: Literature suggests one-sided label smoothing (real label = 0.9), we were doing two-sided earlier (real 0.9, fake 0.1). Repeat of #14 with
--one-sided label smoothing gave 1,3 and 2,1 as the standout performers.
Test #16: Varied depth with slower learning rate and 0.0001 L2 regularization term. Inconclusive.

__VARYING OTHER PARAMETERS__

Test #17: Varying L2 regularization terms, 0, 0.001, 0.002, with 2-hidden networks. No pattern emerging.
Test #18: Trying with a wider range to see if a pattern emerges. Repeat of #17 with 0,0.003,0.006 L2 terms. Higher appeared worse.

Test #19: Varying Adamax beta values 0.5,0.7,0.9. Inconclusive.
Test #20: Varying learning rate, 0.001, 0.002, 0.003. Typo here meant that only generator varied. More appeared better, but should still repeat correctly.
Test #21: As #20 but run correctly. Not conclusive.
Test #22: 0.002/4/6 for the generator and 0.001/2/3 for the dsicriminator, with two internal layers of 100. Inconclusive.
Test #23: Inverted 0.001/2/3 gen and 0.002/4/6 disc. No clear outliers.
Test #24-25: .002/4/6 LRs for both, 2:2 ratio. Inconclusive. Mistakenly put the fixed seed outside the loop, and did not produce reliable results.

__THE FIXED SEED ERA__

Test #26: Varying layer sizes, 30/60. LR pegged to 0.002, and varying whether relativistic loss used. Inconclusive.

Re-considering metrics: Calculating quality metrics during the run slows things down, but saves on having to store large sets of words.

Test #27: Stopped short. Method of producing varying size networks was flawed and messy.

__IMPLEMENTED v2 ENCODING__

Test #28: Re-considered with just 100,100 and 200,200. Relativistic loss looks slightly better than traditional loss, v2 encoding seems to work dramatically better.
Test #29: 100,100 and 200,200 with normal v. relativistic using v2 encoding. Relativistic GAN appears marginally better,
Test #30: Experimenting with WGAN-GP: Extremely poor performance and some discontinuity in this test. Not sure how that happened.

Test #31: Attempted to vary depth and breadth together. More of both seems better, with earlier layers' width mattering slightly more than later layers.

Facepalm moment: Controlling for size when generating the dataset meant that only len 7 words were considered, and the network was not trained to make
--words of varying length. Revised.

Test #32: Bigger initial layer seems better for discriminator, but smaller seems better for generator?
Test #33: Similar to test #32 but showed bigger initial layers worked better for both.

Single run adding some dropout: distribution of specific letters and v-c distribution are correlated (as well they might be), consider combining.

Test #34: Comparing dropout 0/.2/.4 for both paramters. High generator dropout did very well, no dropout at all did badly, but 0/.2 did very well.
Test #35: Tried .3/.6/.9 dropout, higher dropout seemed to perform better.
Test #36: Couldn't reproduce test #34-35, but the specific parameters were not noted.
Test #37: Reproduced run 36, higher dropout performed worse.
Test #38: Scaling back to 0/.2/.4 dropout. Inconclusive.
Test #39: Dropout in the generator created mode collapse. Unsure why.

__THEORY 2__: Generator dropout should be low and limited to the first hidden layer, dropout at other places seems to cause mode collapse.

Test #40: Inconclusive.
Test #41: Varying activation functions (elu, relu, leaky) with one layer at d.4 in both generator and discriminator.
--Leaky relu slight improvement, elu significant improvement.
Test #42: Comparing dropout rates in one layer and activation functions. Confirmed superiority of elu, and higher dropout worked better.

Test #43: Increased mbd to 16, moved back to two hidden layers with dropout on both. Varied dropout in both layers. Inconclusive.
Test #44: One generator dropout and two discriminator dropout. Inconclusive.
Test #45: More dropout seems better, but not uniformly better.
Test #46: Rescaled dropout to test .2,.4,.6,.8. Still inconclusive.
Test #47: Varying size again with dropout this time. Smaller size seemed better on this run.

From 46 and 47: size 50g/100d with .5/.5 dropout seems to be the best setup.

__Hypothesis 1__: If we use sizes 50/100 and vary dropout we'll see better performance than on the size 200/200 runs.

Test #48: size 50/100 with dropout .5/.6/.7/.8: .5/.5 was, in fact, a strong performer with an average oqm of .545 (compared to .614 for the best of run 46)
--Some words from run 48: Barqi, rqmrmb, ajeene, diraw, famresd, chilaa, parund, s mni, ramafel, mmgvmee. 7/10 reasonable. Hypothesis 1 supported.

Single run: 300 epochs on a big six-layer setup got to a last-100 oqm of .550.

Test #49: Varying depth again, 2/3/4/5, with width 75. Lower generator depth was consistently better.
Test #50: Trying a binary variation of depth 3,4 and width 50,100. Lower depth, higher width seemed better, but peak last-half oqm was .707, not great.
Test #51: Moving the variation points further apart, 2,4 and 100,200 with batch size 200. Improved performance on average from wider/shallower architecture here.

Single run: Long, wide, shallow overnight run. 1700d/1900g, 2/2, d3g/d5d. Last-100 oqm peaked at 0.4.
Still produced special characters/unprononuncable gibberish.

Considering what metrics we're looking at. Letter group distro, actual vcsp patterns (e.g. "fish"->"cvcc" rather than just +3 to c, +1 to v) are
theoretically more powerful, but also would slow down the run somewhat.
Created a long-run function in the network that only does the eval step every few epochs.

Test #52: Depth search: 2-8 hidden layers showed weaker performance deeper in. Don't have the setup to try progressively deepening the network at the moment.
Test #53: Tried Adamax betas .5/.6/.7. Inconclusive.
Test #54: Adamax betas .3,.4. Inconclusive.
Test #55: LR .002, .003: Higher better.
Test #56: LR .004, .005: Lower better.

Long overnight run, let's see if the sweet spot remains in the middle.
Test #57: LR .002-.005 for 500 runs. .004 seems to have been the standout.

Long, wide, shallow run at .5 beta and .004 LR got below .33 at the end.

Network architecture arrived on by this point:
Generator: Two hidden layers of width 500, d2 in the first.
Discriminator: Two hidden layers of width 750, d4 in the first.
Both: ELU, Adamax with beta .5 and lr .4.
Seems to be able to produce reasonable results in ~100 epochs.

Next testing phase: Try on multiple datasets. Produced Pokemon v. English dataset comparison and passed to human test subjects to see if they could
tell them apart, results comparable to those produced by chance, so not working in that way right here. Considering other ways to curate datasets.

Runs 58-59: Ratio tests, 1-4. 2:1 seems to be the standout here, which is somewhat surprising, I've been using 1:2 or 1:1 in most places.

'''

'''
__ARSRETE ERA__

2/29/24: Question regarding networks: Running the eval code while the network is running is slow, but if we could save words generated at
each epoch and run the eval code on it later we could do network runs much faster and still evaluate the results. The problem, though, is that we don't want
to have to save massive amounts of random gibberish if we can avoid it, particularly if we're doing a thousand words per batch, 5-10 batches per run...
So: Using the code in ArsRete_Datasets right now, how many words do you need for the various quality metrics to be a good approximation of the whole?
By empirical observation around 200 gets within 5% of 1000.

2/29/24: I'd abandoned WGAN-GP as unworkable before, it seemed to perform very randomly and was highly prone to mode collapse, but I'm revisiting the idea
now. I turned up a StackExchange answer suggesting that WGAN isn't a drop-in replacement for another cost function using the same architecture, since
the math relies on a very effective discriminator where the generator can still learn from even a very good discriminator an effective WGAN-GP will
have a more powerful discriminator network and a discriminator/generator training ratio weighted towards the discriminator. I'd also attempted to run the
WGAN-GP using the same minibatch-discriminator that worked so well in preventing mode collapse in a normal BCEL network; it isn't exactly the same as batchnorm,
but the literature advises avoiding batchnorm in WGAN-GP, particularly in the discriminator, and MBD shares batchnorm's property of making samples dependent on
each other. I had a breakthrough fiddling with some parameters today where applying layernorm, the suggested batchnorm alternative, to every layer of both
generator and discriminator with large batches (300, in the last test) manages to avoid mode collapse for a WGAN-GP over a short run. It didn't work
forever, mode collapse still kicked in around epoch 17, but that's better than we'd seen before. Upping batch size to 500 kicked it out to around 50.

2/29/24: Setting up for an overnight parameter search run the question becomes what parameters are we searching for? Whatever parameters will make the
WGAN-GP network perform better, of course, but do we know what those are? The February parameter search runs aren't necessarily helpful here, they were
all for classic BCEL networks, so we're kind of starting from scratch here, using WGAN-GP.
Experimental Design:
--Searching for: Longest run with no mode collapse (defined as "any two words are identical").
--What variables can we tune?
---Shape (depth, width)
---Activation functions
---Optimizer type and parameters
---Dropout
---G/D training ratios

We can't run all of those right now, and even if we could that doesn't necessarily help. Let's try:
-Fixed shape: 2x150 generator, 3x200 discriminator
-Fixed ratios: 3 discriminator:1 generator
-No dropout
-Activation functions and optimizers: Review of WGAN-GP implementations on Github suggests Adam optimizer and either L2 or simple R activations.
-We have, in theory, five implemented optimizers to consider. Let's try and do the Pytorch default arguments for all five.
-We also have several activation functions to look at, though the demands of time may keep us from running them all overnight here.
-If this doesn't run significantly faster as a result of being narrower doing 5x5x5x5 here could be a fifty-hour run. Let's drop back on our ambitions a bit.
-With just two parameters set to run for 200 epochs this looks like we're looking at a 2.5hr run.  Could we do longer? Hrm. Tests.
-Still running for 200 epochs, but this time we're setting for size 250 gen/500 disc, and have batches of 500.
-With 2.5-3s per run, average to 2.75, we're looking at 550s-ish (<10m) per parameter set, around four hours.
-If this all holds we should be done around 9:30 tonight, and if it doesn't hold it probably won't blow up massively and still be running in the morning. Woo!
-Testing activation functions only here. If we get conclusive results can try testing optimizers later.

3/1/24: WGAN-GP test search 1 last night ran about 7m per run, down below 2s per epoch when nothing else was being done with the computer.
Results!
How do we define mode collapse? Nothing ever collapsed completely, but some ended up with highly restricted word sets.
In theory mode collapse means "the optimizer gets stuck in a loop". Can we test for that?
Number of uniques/number of patterns groupings are uniform for fixed loop a (generator activation function).:
0 (elu): uniques 200-200, patterns 199
1 (sigmoid): uniques 16-34, patterns 197
2 (tanh): uniques 79-110, patterns 199
3 (relu): uniques 199-200, patterns 199
4 (leaky): uniques 199-200, patterns 199

So: Mode collapse with WGAN-GP is avoidable.

Next: What of metrics for our WGAN-GP test?
#Problem: Unfeasably long eval runs for use during normal workday (almost 6 minutes per, is 2.5hrs of eval run)
#I don't know if I want to be looking at the eval at 12:50 today, so let's see which element takes the longest...CVC is taking 1.6s per, skip that
Without CVC we're looking at 2.5-3s to evaluate a whole 200-epoch run, which is much better.

Graph-wise metrics don't really change across epochs here. Some evaluation functions are better than others, but this isn't learning at all.
Checking through network design. It appears that I can make a WGAN turn random noise into different random noise, but not into words.

3/4/24: Couple of quick initial runs with larger networks have produced letter distributions that don't appear to be immediately diverging,
even if the results aren't great. Setting up for a 5x5 optimizer test: Adam, Adagrad, Adamax, RMSprop, and SGD on gen/disc with default arguments.
The question of why the cost function pegs at 2*GPW for the discriminator and 0 or -1 for the generator remains to be considered.
#Test run stopped short, under test conditions different optimizers were producing identical outputs

#Next test: 3x3 run, all Adam, looking at LR .001/.002/.003 for gen/disc
#Changing learning rates still produces identical outputs.
#Something is EXTREMELY wrong with this implementation, and I don't know what.

3/5/24: Comparing model change between BCEL and WGAN over 100 runs (%0s in weights/biases, total and abs total delta), not a lot to set them apart. On paper it looks like the WGAN
should be working, it's just not. Consider pre-training the discriminator?
#Adjusted real and fake labels to account for 1 or 2-size labels, but that isn't impacting WGAN, no labels there.
Attempting to diagnose the problem. Tried pre-training the discriminator, it appears that the discriminator just isn't learning right now, it drops from .6-ish to bounce around between
.45 and .55 accuracy. Flipping the sign of discriminator error: Stays higher for longer, but still falls off. Hrm. Train multiple times per subset?
No apparent effect from 2,0. Let's try reintroducing a bit of dropout. More stable with d4, still not great performance. Drop LR from .003 to .002? Still bouncing around a lot, let's try
dropping LR more. Dropping below .002 doesn't seem to have much increased impact. ADAM betas?
Width didn't make that much difference (750 and 1750 both worse than 1250), depth didn't help (3 hidden worse than 2 hidden), 3 hidden with 3 dropout did't help either.
Further experimentation: On to trying RMSprop. Still highly variable, but slightly better results. No idea why this network won't train at all on 3 hidden layers.

3/6/24: Have tuned a model where the discriminator can learn the dataset most of the time, still collapsing some, so let's dig into the generator parameters.
First test of the day: 25x3 pre-train runs, 1,250 generator size. Pre-train failed a bit (dropped to accuracy .3), let's see if it works.
#Relatively quick mode collapse. Hrm. Pre-train more?
#Still not great. Performance in theory relies on a competent discriminator, but this one is somewhat unreliable.
Trying better weight initialization (Xavier_normal for sigmoid/tanh, Kaiming_normal for elu/relu)
Accuracy remained more stable, still started drifting into mode collapse. Next try: Reduced weight penalties.
Pre-training on reduced weight penalties is showing some promise, stably holding above .6.
With better weight initialization and reduced gradient penalty we got something that looked like it was improving over 100 runs. Final wasn't great, but still. Chase this.

Let's start formal run numbering for the new era!

Run 1: Single run. WGAN GP1, M8/l reg, 1250/1250 g, 1250/1250 d, g .002/.5 Adam, d .006/.4 Adam. Pretrain 35x5, run 100 at 6:1. Graph indicated learning over time.

Run 2: Single run. WGAN GP1, M8/l reg, 1250/1250 both. g .003/.5 Adam, d .003/.3 Adam. Pretrain 35x5, run 100 at 4:1. Graph indicated no learning over time.

Run 3: Let's try varying one variable at a time. Return to stats of run 1 and see if that still indicates success or if that was a fluke.
Current runs are not fixed seed, so randomness does come into it.
During-run observations: Pretrain started .638, dropped off to .520 at step 4, increased intermittently from there. 6s-ish per step. Improved past strating value by step 14, but fluctuated around 0.6 from then.
Interesting that error's fixed stable value is 1.0322, when we ran with gp10 the error was 10.62-ish, so lower seems to produce a more reasonable signal rather than just squishing all weights down very small?
I wonder what about the gp algorithm is producing a value always so close to 1?
Discriminator accuracy is growing during the run, started around .63 but has grown to .67-.7 by step 10
Discriminator accuracy stayed between .6 and .7 for the majority of the run.
Final performance was intermittent, but did move.
Randomness and relatively short run may have influenced this. Let's try dropping batch size again and seeing what happens.

Run 4: Same parameters as 1,3 but with batch size 20. In theory this should make the data noisier, but better overall. Initial pretraining not showing promise, didn't get about .5 until step 4.
Not ideal performance on pretraining, got as high as 0.58 but below 0.5 at the end.
Batch size slowdown is considerable, 11-12s with batch size 20 where it was 5-6s with size 32. If this doesn't produce improved results I may go back and try larger batches again.
Generator started winning around step 23 here, and kept winning.
Result: Irregular, but not terrible.
Let's bump batch size up some, and maybe consider fiddling with dropout in run 6.

Run 5: Batch size 64, and bumped Adam betas up to .6
Very strong pre-training. Up above .8 by the end, and faster overall, 5-5.5s-ish per epoch.
During run da peaked in the 0.85 range, but 0.75-0.8 was more common. Doesn't seem to have translated into great gen performance. 3.71 ld at the end.
Word output seems to have actively got worse here.

Hrm

Run 6 (no dropout) and run 7 (d2 on all layers) both mode-collapsed quite badly. In the BCEL networks having dropout on two generator layers essentially killed the run, let's see if
we can do dropout on one generator and more discriminator layers, see what that does.

Run 8: d4 on all discriminator layers and generator layer 1: Suddenly humongus discriminator error? But the accuracy doesn't seem to be suffering. ??
Would have been super funny if that somehow worked really well, but no, it collapsed like the rest.

Run 9: Removing d4 from output discriminator layer and changing signs in algorithm to match original paper, see what that does.
Gen pretraining still bouncing around a bit, but error looks more stable. Pretrain finished at .63. Run was largely in the range .55-.65.
Did okay, then fell off a cliff. Mode collapse remains a persistent issue.

Run 10: Taking out MBD in favor of layernorm. Will that make things better, or worse, wrt collapse?
Run was descending nicely, and then at epoch 33 it just crashed. Hrm. Does that signify the moment of mode collapese? Consider.

Run 11: Dropping off the dl1 dropout.
Decent evidence of learning! Will fiddle more from this starting point.

Run 12: Next experiment is to bump GPW up from 1 to 3 and see what that does.
Certainly seems to have improved discriminator stability. Pushing into the .8-.85 range pretty fast during the main run.
Improved briefly, but then spiked out. 

Run 13: Reverting GPW to 1, see if 11 is repeatable.
More stable, but also dips, then rises.

Run 14: Can we do GPW0.5? Yup! What does it do? Not much! Some areas of learning, some areas of not learning

Run 15: What if we had no gradient penalty? What does that look like?
Not unlike with GP. If no-GP and with-GP are so hard to tell apart it may be that my GP implementation doesn't work.

Run 16: Tried copy-pasting someone else's GP implementation, didn't seem to work.

3/7/24: Attempting WC instead of GP. Review of literature suggests pre-training isn't important.

Run 17: Attempting WC1 to see what happens. Certainly faster than GP. Will it be more stable?
Basically unchanged in character from the GP examples (jumps around without really trending up or down). Consider: Deeper?

Run 18: Collapse almost from day 0. Let's try reintroducing MBD reg, see what happens.
Significantly increases run time (6s, up from 3s per epoch).
Doesn't improve results.

Run 19: Internet advice suggests RMSprop for WGAN-CP rather than Adam, see what that does?
Interestingly runs got longer, started around 7s, got up to 10s, back down to 8s at run 17.
DA is relatively stable in the .6-.63 range.
Never really got better, but doesn't appear to have collapsed.

Run 20: Slowed LR down to 0.00005. Curve went up, then down. Would it keep going down over a longer run?

Run 21: Longer run of run 20, over 300 rather than 100. No improvement, still just bouncing around.

I think for the moment I'm going to have to abandon the WGAN, it doesn't seem to work for this data.

Run 22: BCE network with same structure as run 20. Hit mode collapse pretty fast. Earlier experimentation suggested this data doesn't like nets more than 2 hidden layers deep.
Got stuck in some kind of static equilibrium and wouldn't learn until I went back to size 2 disc out

Runs 23-25: I'm completely baffled, BCE parameters similar to the old one just not working here for some reason.

Run 26: What. The. Hell. Maybe something other than the WGAN not working was making this not work.

Run 27: Size 1 disc out (correct for old version) no longer producing static output, which is good news. It'd be better news if it produced output that worked.
(BCEL is supposed to be used with size 1 output...) During run: error isn't changing much, but accuracy is slowly climbing from .25 to .35

Run 28: Graph looks not unlike we're maximizing where we should be minimizing. If we flip the sign on the error what happens? This probably won't help, it's not wrong in the original
version. Don't think it's helping, error and discriminator accuracy are both much worse and staying that way.
Should be basically the same, but the old one is working and this one isn't.
It's not the text encodings, we're reading from the same wordset...Batch size? Doesn't seem to make a difference. Cost functions, activation functions, nothing seems to make this converge.
Hrm.
Hrrrrrrrrrrrrrrrrrm.
Added the outer folder to the Path, what happens if we import that GAN object and try running it instead of ARM?
Using the parameters of the last big single run in the ATG version. Currently not set up to give it the in-run set comparisons, but error, at least, seems to be moving around a lot.
I wonder what the actual screw-up was?
I guess we'll see what the actual performance here is in a bit.

Using vdist only (most predictable). Going from la only to la+lb, plus batch size 32, we're seeing vm (compared using ATG compare) bouncing around, but we don't have a good comparison.
since ATG compare used size 36 vectors to store size 32 frequency numbers, and is thus producing "better" numbers for the same result.
Ideally I'd get ATG wired up to use ARM encode/decode, but at the moment...hrm. Tests. Regardless, it isn't blowing up or collapsing like the ARM network, at least in this short run, maybe sit
down and re-refactor.

Run 29: HOLY SHIT IT'S WORKING, we're not pegged to DE ~= GPW
Generator error and discriminator error crashed badly after epoch 2, but started to come back after epoch 22, if we can make the dump tool run I'd be fascinated to see some patterns here.

Run 30: Short test of two epochs, worked itself into mode collapse and then back out.

Run 31: IT WORKS! BEAUTIFULLY! Lfreq went from 6.4 to 1.34 in the first epoch, 1.2 to 0.8 in the second, and just kept getting better! Some hyperparameter searching to go, but this is a
breakthrough of epic proportions! Last 5 words: riulik, wikpba, ekveqj, makver, lakru. Strong performance! Of particular note: Markov parameter improves overall as well.
Accuracy number may not be a relevant metric for WGAN, it drops to 0 pretty fast and mostly stays there.

#Run 32: Longer version of run 31. Generator error blowing up some, but coming back, and we're seeing continung progress (e35 - V0.25)
Hrm. Considering gen-err blowing up I wonder if there's a GP-tuning solution there. Come on! I want to see M2 break 0.3! 0.2951, 0.2955...

3/12/24: WGAN working, but peak performance on the overnight runs still short of ideal.

Run 33: Increased to three layers deep. Slow run, unstable Vm, non-great answers.

Run 34: Removed dropout from one g layer, better performance (down to 0.2 Lm by the end of 100 runs)

Still slow.

3/13/24: The 3/8 overnight runs were in the neighborhood of 3hr45m for 1,500 epochs, but got pretty much to peak performance by around 500.

Run 35: Trying parameter search for run ratio, 1:1 is going quickly but also seems to be evincing the older version's issue with flat error and no performance improvement.
Fascinating. Parameter search is one variable, network size is another change from the last run. I'm going to stop the parameter search run manually and do a few even shorter runs.
Alternate theory: Too much layernorm? Single-run has it in all layers but the last, this one has it in the last layer as well.
Graphs show learning early, then saturate. Run again with more a? 2:1 through 7:1

Run 36: No standouts in the 2:1 through 7:1 test. Culprit is likely either too much layernorm or width.

Run 37: Repeat run 36 without last-layer layernorm. Seems contributory, seeing learning in run prints, graph holds with that theory, more Disc runs better and no l in final layer makes this work.

Run 38: Changing ratio range from 2-7 to 5-10 and reducing layernorm to middle two disc layers. With reduced layernorm showed strong performance, with more disc to gen runs still better.
Considering absolute performance tradeoff of ratio. Let's try a run with no layernorm at all. (Vm got to the .6-.4 range by the end.)

Run 39: With ratios 8,10,12,14,16,18 we're seeing some diminishing returns from bigger sets, 12 and 14 are looking relatively stable. Lack of layernorm seems to have not really affected output.
Ended with VM in the .5-.3 range.

Run 40: Let's try changing lr instead of ratios, see if we can get similar performance faster. Checking from .005-.008. In the 20-run timeframe .006-.008 clustered together and .005 was slightly above.
Let's try changing LR and ratios and see if the combination helps.
Averages for parameter 0 (.005, .006, .007 disc lr) show separation with lower = better below 100 batches, but rapidly converge from that point to become indistinguishable by 150.
Parameter 1 (ratio 6:1, 8:1, 10:1) show more reliable separation with lower = better for the broad length of the run, finishing around .1 apart (.45 for 6:1, .36 for 10:1)
Implication being that LR doesn't have much impact in the longer range, which WGAN papers suggest as an advantage of minimizing necessary hypertuning.

Run 41: Minimization of hypertuning in a WGAN-GP environment means that we probably won't need to fiddle with LR further, will peg to 0.005 for symmetry. Network geometry becomes the next battleground.
Depth, breadth, both? Let's fix ratio at 8:1 and try depth on narrow networks.

Run 42: Pegged LR to 0.005, ratio to 8:1. Dropout .3g/.5d is always in 2nd layer, no reg, elu activations. 100 width g, 150 width d. 2,3,4 hidden layers for both networks. Also running to 30 epochs this time.
Can we usefully quantify time difference? Hard to say. Shouldn't change much over the length of the run, at least.
R42 quantities:
Time: Cost of time for adding a layer appears proportional to size of layer, ~10% for adding a size 100 layer, ~15% for adding a size 150 layer.
Size 100 (G) layers time/batch for 2,3,4: .0952, .0993, .1095
Size 150 (D) layers time/batch for 2,3,4: .0897, .0986, .1157
Benefits of adding a layer?
More G layers: Graph starts further apart, but converges some as they go down. No clear answer as to what's going to be better in the long run, all pretty close at the end.
I might need to introduce a 2nd-half metric to avoid the inflationary effect of the early peaks here.
On last-half metrics: Averages pretty close: .4340, .4684, .4526 for 2/3/4 layers. Consider extending or trying with a different seed to see if the relationship is random or if 3 is actually worse than both 2 and 4.
More D layers: Last-half measure shows graph that's pretty close together all the way through, but 4 hidden layers is consistently worse in the last half: .429, .410, .515 for 2/3/4 layers.

Fascinating! Will repeat with different manual seed and see if the relationships (gen best: 4, 2, 3; disc best: 3, 2, 4) still hold or if this is largely random.

Run 43: Repeat of run 42 with manual_seed(2) instead of manual_seed(0) to see whether relationships are consistent across seeds. If conclusive could run a third manual_seed to confirm, or
move on to width testing. Will also probably want to consider more than 4 hidden layers and see what happens.
Literature suggests that in general more discriminator depth and lower generator depth is the best case.
Info from Biau, Sangnier, and Tanielian (https://www.jmlr.org/papers/volume22/20-553/20-553.pdf pg.22-23) suggests that for a given generator depth incrementing discriminator depth generally makes the model stronger,
and for a given discriminator depth incrementing generator depth generally makes the model weaker.

Using last-half data from run 43:
G-depth 2,3,4 to lfreq: .429, .463, .464 avg.
D-depth 2,3,4 to lfreq: .432, .437, .487 avg.

In this test it appears that we're seeing depth making the model weaker for both g and d. I think we probably need to look at the overall grid.

Grid for run 42:

Best: (gen depth y, disc depth x)
|4| 0.2527 0.2089 0.3293 ||
|3| 0.2805 0.1897 0.339  ||
|2| 0.2644 0.2568 0.2897 ||
    2      3      4     

Avg: (gen depth y, disc depth x)
|4| 0.4407 0.4022 0.5152 ||
|3| 0.4439 0.4078 0.5537 ||
|2| 0.404  0.4216 0.4764 ||
    2      3      4     

Grid for run 43:

Best: (gen depth y, disc depth x)
|4| 0.255  0.2515 0.2785 ||
|3| 0.2445 0.2014 0.3122 ||
|2| 0.2868 0.2644 0.291  ||
    2      3      4     

Avg: (gen depth y, disc depth x)
|4| 0.4338 0.4233 0.5364 ||
|3| 0.4227 0.464  0.5024 ||
|2| 0.4413 0.4239 0.4246 ||
    2      3      4     

Overall grid for both isn't massively more enlightening. Relationships are somewhat inconsistent and contradictory. Would averaging between the two seeds present a pattern?

Averaged average table:
.4372 .4127 .5258
.4333 .4359 .5280
.4226 .4227 .4505

Does the y=-x pattern manifest here? We have 12 transitions, and it holds for 5 of them, which isn't promising. Factor in the 6 double-transitions and it holds for 7 of 18.

So we have not experimentally verified Biau, Sangnier, and Tanielian, but we also haven't contradicted them.
The general rule on WGANs is that the discriminator's power is more important to making it work properly than the generator's, though, and the results on ratio/LR hold with that.
Let's do a longer test run on 2 hidden gen/5 hidden disc, see how we do.
15s per epoch at 750/1500 with this depth, but parameters are going in the right direction. 1,500s is, what, 25min?
Around epoch 50 seeing 20-25s/epoch. Around .2 letter, .65 m1, .29 m2
Around epoch 80 still 20-25s/epoch, around .15 letter, .68 m1, .3 m2. Slowed down a lot by this point. Don't think the error graph is going to tell us much, but the small DE magnitude
and big GE magnitude makes me wonder if it's possible for D to be strong enough to overwhelm G by the later points, if you can break out of D overwhelming G near the beginning.
Fascinatingly GE magnitude dropped right near the end and produced a pretty great result (.12/.70/.33).

Probably a good idea to test some of this on the overnight run, if I have time to set it up.

Single run testing revised Markov score. Intent was to make something where the distance was scaled to 1=random/0=reference like in the v-metric, not 100% whether it worked
Presently looking at v.1544, M1 2.056, M2 1.675 at epoch 57
Also possible that the generator is just honestly that much worse than random.
It is improving, but I kind of want to see the raw scores, now.
This configuration (4 hidden D, 2 hidden g) with an 8:1 run ratio is performing somewhat better than the 6:1 ratio with 5/2 hidden layers.

3/14/24

Run 44: Overnight depth test, 500 runs for 4 hidden D/2 hidden G, 8:1.
Gen 100/200/300: Last-half avg lfreq: .1486, .1582, .1753. Graph a bit hard to tell apart.
Disc 250/450/650: Last-half avg lfreq: .1417, .1635, .1768. Fascinating.
In both cases smaller networks did better over a long run.
Let's look at 0 (100/250)'s words vs. 8 (300/650).

0: Last 10: ref ded, gorikds, seogedr, eoocin, vepmwry, am0er, accitms, belfas, saamil, woutdmd. Not great, not terrible.
8: Last 10: fetymep, squpil-, eeuer, geleqnt, rehymds, baldice, unfile, dmmlme, cadni, migalds. Comparably ungreat.

Hrm. Not sure how to read this. Can we isolate best runs?
Hrm. Minimum lfreq isn't a great idea, the best lfreq epochs seem to have a lot of random unpronouncable combinations of vowels.
Funnily enough that's also the minimum Markov1 score.
Maximum Markov1 score, though? Also lots of random combinations of vowels.
For pronouncability of set as a whole consider: metaphone index (slow-ish to compute), if a word has a metaphone value that's also in the reference set that suggests it's relatively pronouncable.

In theory we now have a pronouncability score determined based on metaphone and vowel-consonant pattern, we could train a network to predict that score instead of computing it directly?
...I don't know if I'm a moron or a genius.
Classifier network set up to predict pronouncability score, since computing it directly is computationally intensive. Peaking around 75% accuracy on binary good/bad in the space of 100 runs,
overnight run is set up with best config to go for 1,000. If it works I'll start working on how to save/load models tomorrow.

3/15/24
Classifier effectiveness weirdly cyclical, peaked in the .77-.79 range for most of the run, but crashed to 0.6 or 0.4 for about fifty epochs a couple of times.
Probably a good idea to try and work through short runs a bit more in search of better generalization accuracy
Hrm. Changing over to BCEL on 1-or-0 seems to have worked wonders (100% eval accuracy inside 12 epochs). Accurate result, or bug?
Bug. Fixed. Next question: Deeper, narrower models seem to make the classifier here do better, what happens if we try deeper, narrower WGAN?

Single test results: Interesting. Peak performance is comparable to shallow, wide networks, only faster, and values bounce around a lot more.

Weekend/overnight run will probably be a test of ever deeper networks.
Single test: Extreme case of depth 5/25 bounced in and out of mode collapse for a while initially, but seems to be trending better going forward in an unstable and bouncing-around way
I wonder what would happen if we tried this massive depth with a greater GPW?
GPW25: Error still blowing up out of control (magnitude in the hundreds), but results surprisingly not terrible (v sometimes less than 2!)
GPW35: Stuck in mode collapse for longer. Not great.

Run 45: Overnight run, with gen depth 3,4,5 and disc depth 10,12,14,16,18,20,22,24,26,28
This may be a dead end, but it'd be really cool if one of these turned out to be a magic combination of depths that gave good results first time, every time.

3/18/24
Looks like we made it through three of run 45 (3:10, 3:12, 3:14) before being shut off for updates.
Lots of judderiness in the early runs, around halfway through 3:10 settled in around .22 and 3:12 settled in around .12, though more irregular, while 3:14 bounced between them and occasionally spiked
Last-half avg (includes before the equilibrium state): 3:10 .305, 3:12 .105, 3:14 .215
Shorter runs than 2.5hrs may not help, given that we hit the equilibrium state around batch 9,000.
Next question: What do the pronouncability scores look like? May have broke something attempting to graph it. Because I was attmepting a 5-d graph, like a moron. Thbbt.
Graph indicates that after initial jumpiness score 0-1 goes down and score 3 goes up, with score 2 remaining the most common and score 4 rising slightly, but still a very small percentage

Next question: Is there a relationship between these proportions and my other measures? Markov number seems to correlate with the Metaphone-Vodsit score better than Lfreq

Run 46: Narrow version of run 45 (width 25 gen/35 disc), depth 2,3,4 gen, 10,12,14,16,18 disc.
Graph and numeric approach suggest improvements for shallower generator, not by a lot, but .2956, .3079, .3340 does suggest shallower gen did better.
For disc info similarly isn't conclusive, 14,16 did best, but only by some.
Comparative graph of the Markov data is similarly inconclusive.

3/19/24
In theory from run 46 the best individual performers should be 2,14 and 2,16 (numbers 2 and 3).
Quantitative assessment: lowest last-half average LFreqD are 0 (2:10) at .2624, 8 (3:16) at .2777, and 7 (3:14) at .2813
Markov1: Best last-half average are 13 (4:16) at .6056, 8 (3:16) at .6029, and 14 (4:18) at .6022
Markov2: Best last-half average are 14 (4:18) at .2590, 13 (4:16) at .2533, and 11 (4:12) at .2526

Hrm. Fascinating. Let me try throwing these in a color-code grid and take a look.
Actual numerical differences, particularly over 1,000 epoch runs where the variance in any individual dataset is quite small, isn't very conclusive.

3/20/24
Run 47: Overnight run with depths 3 gen, 12 disc.
Testing breadth 25/35/45 gen, 45/65/85 disc.
Graph suggests this begins relatively stable and then starts jumping around
Not sure what that means, but it's worth looking into.
Lower gen breadth seems to jump around less. Quantitatively .35 avg vs. .51 and .44 for the larger two.
Similar jitteriness starting around 3,000 batches for disc breadth. Higher disc breadth starts jumping sooner, but doesn't jump as much.
Borne out by numerical, .30 vs. .48 and .52 for the others.
In theory #6 (lowest gen, highest disc) should be the best by this graph, but it doesn't look like that's the case, 8 (high both) is the best on all parameters.
Interesting that 8 also had the highest gen error by the end. I wonder if GP is enough here?

Doing a bit of manual short-run fiddling, most of the parameters same as in run 47 (3 gen, 12 disc, 8:1 ratio, .005/.6/.99 Adamax)
Let's see if even wider layers (200 gen/500 disc) or l2 reg do something.
About 11s/epoch, to start! Taking a while to dig out of collapse mode. Might take off second disc dropout, see what happens.
...I mean, while it is taking a while to dig out of collapse mode it is doing it, dropped below V1.0 at epoch 22.
Hrm. I also wonder if ballooning error is correlated with ballooning weights. Should see if there's a fast way to print those.
Very much not fast, slowing down run a lot, I note that average gen weight is bouncing around/going down and average disc weight is increasing in the short term.
Are we going to hit an equilibrium, or keep increasing?
Don't seem massively correlated to error, given that disc is the one increasing and disc error is the more controllable.
One thing I think I DO need (evinced by this test) is a better framework for adding and removing test variables to the GAN class.
Tried 500,1500 widths and ended up with 90+s/epoch, not easily doable.
Dropped back to 3/9 layers with widths 200/300. Down to around 6-8s/epoch, collapsing for a while, came back around epoch 24.
Problem: This set bounces around a lot in quality. I kind of want to try and get back to something with performance curves more like the initial 0308 tests, with higher gen lr and shallower, wider nets.
Deep/narrow is cool, but may not be that efficient.
That said...maybe compare them, see what I see?
Gr. Just jumped from V.444 to V.844 at epoch 80.
Comparison function added for quick comparison of n runs along m variables.
Interesting that even when V is jumping a lot M doesn't change that much, unless spiking when we get a temporary collapse on a decent word.

Hrm. Comparison graph suggests shorter/wider is a better overall performer.
Single run at 500x2 gen/1000x3 disc rather than 750x2/1500x2 and bringing MBD back in.
About 8-9s per run, and V, at least, is improving monotonically, although M is still bouncing around.
Interestingly Acc is now doing something.
Also dropped GP back to 5
By epoch 25 we're at V.25-.3, and it's bouncing a bit, but still, solid spot to be in by epoch 25.

Can't now remember why I wanted to try deep/narrow like that earlier. Hrm.
Let's try adding a 4th disc hid and see what emerges.
On these metrics and these time scales d-hid 3 did slightly better than d_hid 4
If we do a full ARD.setpro graph on these what do we see?
Three layers remains slightly better on the graph, no quantitative description, though.

Next: Return to 2,3, and raise gp to 10

I wonder if it's almost time to introduce filtering, and then do the poketest again.
Should probably get human subjects to rate pronouncability and see how well the metrics do.

Difficult to separate, probably needs more work.

3/22/24
Set overnight run for 10,000 iterations, even at 8s/epoch 80,000s is...22 hours, so coming back to find it at only epoch 8,250 not too surprising.
Ballooning error/poor results by the 8,250s, though. I do kind of want to keep it running to see the graph, hopefully it won't take four hours to finish.
Probably not worth it at this point.

3/25/24
Returning to work on this bit for a little.
Did a quick poketest run with width 2/4 and depth 300/750, at around epoch 470 (4-6s per) I'm contemplating cutting it short so I can do a bit of actual work.
We know numbers aren't perfectly indicative, things that look like they might be pokemon ("arulby","utgario","steabid") coexist with things with numbers ("a0ead","cksiog0","p2rxmyg"), even at .2 V/.7 M1.
Will stall a bit longer with the ext report, maybe we'll hold out.
I kind of want to do an overnight run to see if I can find a graph of the ballooning error result from the 3/21-3/22 overnight run
Single pokerun here: Improvement did continue over the full thousand runs, but we got most of the way to peak performance on M1,LF by about 100-125 runs. Interesting that M2 keeps improving while the others stagnate.
Not sure what it means, exactly. Consider proscore graph.
Considering proscore graph "scores at least 2" tracks well with M1, doesn't increase as M2 increases. I wonder if we do the bigger breakdown ("at least n" for n in 0,1,2,3,4) we'll see something that
M2 indicates?
"At least 1" tracks M1 very closely, "at last 2" and "at least 3" increase slowly with M2. 4 continues to go up as well.
Hrm. Indications suggest I should do a really long overnight run tonight.
Secondary question: If I try to speed this up using really narrow runs will performance still go up?
Let's divide by 10 (30/75 depth), see what the comparative performance over 1,000 runs looks like. And the comparative time. 300/750 was over an hour.
Epochs are about 1.2-1.3s this time, rather than 4-6s. 1/10th the width, 1/3rd the run time. Performance? Still above V0.5 by epoch 100, though M1 and M2 are looking okay.
Words looking non-great at the moment, we'll see how we do towards the end.
I'm tempted to hold off on doing proscore, on the logic that it tracks M1/M2 pretty well, but I should probably do proscore on the narrow graph to see whether it still tracks M1/M2 the same way.
Narrow run V around 300 is down around 0.3-0.4, not great. M scores pretty stable.
Hrm. Looking at the moment like peak performance is worse, but we take a similar amount of runs to hit it.
I say that, but approaching 600 we're flirting with a V score below 0.3.
/10 width does show roughly similar patterns with 300/750, M1 tracks well to score 1+, mils increase in 2+/3+ kind of tracks to M2.
Given that, how do these two compare on speed and performance?
This test shows the faster, narrower one halting learning quite early while the slower, wider one keeps going. Not a good fast-run substitute. May expand again.
Hrm. Hrm. Hrm. Would going 10x wider crash something? I know we've done 1,000/1,500 quite happily...let's try it, see how horrible it is. Only doing 100 epochs, just in case.
Assuming it makes it that far.
...Next question: How does epoch run time scale against net width?
Worse than linearly, I think.
...Well, it's been over a minute since I started the run and no output. Maybe try something slightly less massive. What's the run time on 5x?
I wonder if there's an effective upper bound on the width of a single component network?
1000,2000 took 30s for an epoch. Metrics improved a lot from e1 to e2, though. Let's keep it going and see.
Epoch time going up, e4 was 40s. By halfway mostly 43-44s., solid metric performance (.1957V, .6509 M1, .2782 M2). Seem to be relatively stable by here. Interesting that mid-width stabilized later
and then sort of kept going.
Also interesting that with 1000,2000 acc is still nonzero.

500,2000: Might actually be the same speed as 1000,2000 since I haven't seen the latter running with no other computer activity...
Hrm. 30/75's peak performance cap suggests that's definitely a lower bound at which the network can't emulate the characteristics of the input.
Much faster to run the comparisons for a shorter run...
Significantly smoother than the pokedata. Is that the result of replacing 300,750 with 500,2000 or the data? Hard to say.
Hrm. Performance and proscore performance looks very similar, lending further credence to the idea that generator layers don't mean as much.
Comparison data favors the smaller, though.
Hrm hrm hrm hrm hrm. If I can stop dozing off for a minute at a time.

Trying 100,1500 with more d dropout (every hidden layer) to see what happens.
20-25s/epoch here. With the extra dropout comparable performance, but much more regular and seemed to be still sloping down at the end.

3/26/24
100,1500 with m12 and d0.5 after the three middle layers. 1,000 runs at 22s/run went for six hours. How well did it do? I don't see the exploding-error mode in the text file, let's graph this:
Not currently graphing error...
40s/thousand to do proscore means this may be 10-11 minutes to graph.
Just under 10m, good job, computer. Slight upwards slope in M2 and score 2+/score 3+ all the way through, though some sawtooth throughout. Let's try graphing just error, see what we see.
DiscE begins over 10,000, gene begins around -50, both get inside the -4,4 window by epoch 200. DiscE trend stays mostly in the -1.0,-0.6 range, with brief spikes outside that, from then on.
GenE is less regular and has much bigger spikes, but still largely confined to the -4,4 window for the rest of the run.
Hrm. Phenomena not necessarily showing up as we'd expect. I wonder if we'd observe a difference with different GPW (this one ran with 7) in a shorter, narrower run.
To the parameter search function!

Run 48: Back from long single runs to a parameter search! a loop is GPW 7,8,9,10,11 and b loop is discriminator dropout value 0.4,0.5,0.6.
3x100/5x200, 8:1 run ratio, elu activation, the works, 300 epochs a run, .005/.007 adamax caps.
Around 95m for the full run.
Hrm. Not much to separate GPW on the Lfreq, pretty uniform.
DiscE: Pretty flat most of the time, but some pretty extreme spikes. No readily apparent pattern in their emergence.
GenE: Interesting. 0 (GPW7) remains pretty uniform near 0, 1 drops around -17 and slowly climbs back, 3 drops around -17 and stays there, and 2,4 drop below -40 and stay there.
Big crash begins around batch 300, same as the big spikes in DiscE

Comparing dropout: Stayed very uniform up until around batch 375, then spikes. 1 (drop 0.5) most stable, 0.4 worst, 0.6 middle.
I suspect the huge spike from ~520-620 is one outlier.
GenE shows a similar pattern with dropout (stable until 250-300, then drops significantly), but shows a monotonic separation (d0.6 drops least, d0.4 drops most, d0.5 in the middle)

Suspicion: Run again, considering higher dropout and lower GPW

Run 49: Like 48, but testing GPW 3,4,5 and disc dropout .6,.7,.8
No real conclusions from the test. Lower GPW seemed to show error just increasing, but I don't know if that's enough to conclude there's some kind of sweet spot.
Next try: Big, long run with GPW greater than the 7 of last night's big run.

Tried a couple of 500-long 50x3 by 200x6 runs with GP12, showed 0.8 disc dropout slightly weaker and significantly jitterier after about epoch 350
Obvious next step: try 0.3 disc dropout? Test stopped, exploding error.

3/27/24
One big run (1,000 epochs) of 2x150,4x300 with 0.4 dropout on all discriminator layers run last night, finished 1811 (40m after departing work).
Seemed pretty normal/stable for the most part.

Hrm. Comparison of short runs of 50x3/200x6 to 75x2/200x4 and 175x2/400x4 shows the extra layers are a lot less stable
Shifting back to normal relu for a bit from elu, see what happens.
This test of 500 runs (175x2/400x4) shows relu outperforming elu (stronger performance and more stable)
Mostly comparable on proscore graph, though relu shows more improvement in at-least-4

Run 50: GPW 3,5,7 and gen depth 2,3,4, all with regular relu.
Estimate 5hrs for overnight run of nine, ~40m for first suggests 7.5hrs

3/28/24
Run 50 conclusion. 35m/iteration, roughly, started around 4:30 and finished around 9:50.
Observations from lfreq: essentially no difference on axis 0 (gpw), gen depth 2 better faster, 3 got there in the end, 4 worse and less reliable
Observations from Markov1: Essentially no difference on axis 0, gen depth 3 seemed to outpace depth 2 in the second half, though 4 remained behind
Observations from Markov2: No difference on axis 0 again, gen depth 3 outpaced depth 2 at around the same point as in Markov1, both ahead of 4.

Let's try one slightly bigger run with normal relu and depth 3/5
Mildly skeptical of the convergence of this one (size 250/750) given that we're seeing V bouncing up and down as much as 0.2 between epochs. Let's back out and try again with a narrower net.
Almost-end result: Quite good, I've seen .08/.8/.5 performance figures, and words like "tabanid", "canula", "pogonia", "libric", "haskive", "parisca" that are definitely not words but sound like they should be in the last 50.
...Also saw "reloads" three times in the last fifty...
Should try the graph, but should also try doing a comparable pokerun.

Perfectly serviceable output. Maybe it's time to try the human experiment again.

3/29/24
Overnight run of 5,000 using relu, relu died completely at 1527 (error pegged to a constant for the rest of the run), but got to V 0.08/M1 0.84/M2 0.6 before that.
Performance figures show best performance hit around 1,000 epochs, about two hours in.
Hrm.
Presumably the next plan is to work out which anti-dying-relu variant works best with a long elu and long leaky run.
...If I were very clever I'd set these up in parameterSearch
If I were very, VERY clever I'd set these up in parameterSearch correctly...
On symmetric error no weird behaviors of any kind, which surprised me a bit. No weird spikes (well...depth/width a bit low for those), no relu collapse, no nothing.

4/8/24
Run 51: Testing relu, elu, leaky 0.1, and leaky 0.2 on generator and discriminator
Hrm. Overnight runs from 3/29 (running about 1hr per over 2,000 epochs) suggest elu better for discriminator but worse for generator. Fascinating.
Isolating elu/relu versions: 0,1,4,5
0: relu/relu, 1 relu/elu, 4 elu,relu, 5 elu/elu
Hypothetically we should see 1 as worst and 4 as best, with 0,5 in the middle.
On error: Instability significantly worse in 1 and 5, 4 slightly worse than 0.
On lfreq: again showing more instability in 1 and 5, 4 slightly better avg and slightly more stable than 0.
Interestingly while 4 gets where it's going faster 0 ends up in a similar place.
On markov1: 4 a little jumpier but better performance than 0, 5 peaks well but unreliably, 1 poor and unreliable
If this is repeatable we've hit on relu gen/elu disc as a better combination of activation functions!

Some bouncing about in the early run. The ballooning GE concerns me. Hrm. Slightly longer run?
Oh, I am a twit. Relu disc/elu gen is what we're going for here.
Funnily enough that produces much more stable early training loops.

4/10/24
Some individual tests. Removing dropout from gen seems to have helped.
Run 52: Checking dropout under consistent experimental conditions. Not that conclusive. Adding dropout makes the performance of the network overall very slightly more stable and very slightly worse.
May be worth it, may not. Could try again with bigger dropout.

4/12/24
Run 53: Checking with 0/0.25/0.5/0.75 instead of 0/0.2/0.4/0.6. Graph not lining up exactly with quant for some reason, but graph and quant suggest that dropout is indistinguishable and no dropout is slightly better.
No dropout seems better in gen, inconclusive in disc. Less conclusive looking at M1, both gen and disc show evidence of crossing back and forth.
Question of what happens over longer runs remains.

4/18/24
Returning to notes after a few bits of unscientific fiddling.
Run 54: LR 0.003, 0.004, 0.005, 0.006 for both g/d.
On LFreq: Lines definitely separate out for late run, but in the order .004/.003/.005/.006. Not sure how much we can conclude here
LFreq, d-LR: Separates out into .006/.005/.004/.003, but unstably. Averages do line up with that, though (.219 for 0.003, .209 for 0.004, 0.201 for 0.005, .197 for .006)
Markov1, g-LR: Shows a fairly clear stack rank, 0.003 at the top, 0.006 at the bottom. Last-half averages .68, .66, .65, .64.
D-LR harder to separate on Markov1 on the graph, but the averages do tell a story, .670/.667/.664/.657 for .006/.005/.004/.003
Markov2, g-LR: Final ordering shows .003 at the top again, averages .315, .301, .292, .283 for .003/.004/.005/.006
Markov2 not so clear for d-LR, .294, .300, .300, .299 for .003/.004/.005/.006
It looks *generally* like better d-LR/weaker g-LR is helping here.
Let's try a bigger spread and see if the effect persists.

Run 55: LR 0.003, 0.005, 0.007, 0.009 for both g/d
Relatively clear-cut to start off, Lfreq g-LR shows lower is better by .204/.208/.232/.254
Lfreq d-LR: Less clear-cut, .251/.226/.208/.214. May suggest .007 is a sweet spot and it falls off at .009, but the graph shows .007 and .009 crossing over each other.
Markov1 g-LR: Shows same behavior as in run 54, 0.003 rises relatively steadily, while the others peak around epoch 50, crash, and then don't catch all the way up.
Markov1 d-LR: The epoch 50 peak is here, too. Higher seems better in general, but 0.007 and 0.009 are very close to each other here.
Markov2 g-LR: Fairly dramatic distinctions here, replicating patterns from Lfreq. .311/.290/.281/.273 stack rank.
Markov2 d-LR: Similar to Markov1, higher is better.

4/22/24
10,000 epoch run takes a while to load, 70-80s per 5%, which is about 25m. Or 24m52s, as it turns out, solid call there.
Long overnight run confirms that the system continues improving over the long term. Approach to asymptotes (m1 0.85, m2 0.65, l 0.06), definitely, but showed continual improvement past the 1,000-epoch mark.
This version additionally showed no large error spikes.





'''

import ArsRete_Datasets as ARD
import ArsRete_Modules as ARM
import ArsRete_Postprocess as ARP
import torch
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
from time import time
from metaphone import doublemetaphone as dm
import json

from ArsRete_Model import GANModel,Classifier,GanFromFile

def groupeval(words,gibberish,names,numbers,numtot):
    
    individuals = []
    s = time()
    outfile = open("ARouttest.txt","w")
    for a in range(numtot):
        datalist = [ARD.Compare(b,words,gibberish) for b in ARP.loadDump(names.format(a))]
        #Transpose:
        dlt = [[datalist[b][c] for b in range(len(datalist))] for c in range(len(datalist[0]))]
        #Write:
        outfile.write("Run Index {}:\n")
        for b in range(len(dlt)):
            outfile.write("M{}:".format(b) + "|".join([str(c) for c in dlt[b]]) + "\n")
        if a != (numtot - 1): outfile.write("\n")
        #Append:
        individuals.append(dlt)
        print("{}: {}".format(a,time()-s))
        s = time()
    outfile.close()
    #We have, in theory, transposed this

def graphGroup(groupdata):
    #Graph groups
    
    for a in range(len(groupdata[0][0][0])):
        #For each TYPE OF DATA:
        for b in groupdata:
            #For each run index:
            for c in groupdata[b]:
                #For each value at that run index:
                localgroups = [groupdata[b][c][d][a] for d in range(len(groupdata[b][c]))]
                subdata = ARP.lineAvg(localgroups)
                plt.plot(subdata,label = "D{}.I{}.V{}".format(a,b,c))
            plt.legend()
            plt.show()

def quickGraph(in_file,ref = False,errmode = False):
    t1 = ARP.loadNewDump(in_file)
    if errmode:
        plt.plot(ARP.trendLine(t1["DiscE"][0],100),label = "DiscE")
        plt.plot(ARP.trendLine(t1["GenE"][0],100),label = "GenE")
    plt.plot(ARP.trendLine(t1["LFreqD"][0],100),label = "LF")
    plt.plot(ARP.trendLine(t1["Markov1"][0],100),label = "M1")
    plt.plot(ARP.trendLine(t1["Markov2"][0],100),label = "M2")
    start = time()
    last = time()
    if ref:
        perfd = [[],[],[],[],[]]
        concatl = [a for b in t1["Words"] for a in b]
        for a in range(len(concatl)):
            temp = ARD.prodistro(concatl[a],ref)
            for b in range(5): perfd[b].append(temp[1][b])
            if time() - last > 10: 
                print(a, time() - last, ARP.progressBar(a/len(concatl)))
                last = time()
        for a in range(5):
            plt.plot(ARP.trendLine(perfd[a],100),label = "Score at least {}".format(a))
        print("Perfstep done in {}".format(time() - start))
    
    plt.legend()
    plt.show()

def quickCompare(files,parameters):
    quantl = []
    
    for a in range(len(files)):
        tn = ARP.loadNewDump(files[a])
        tq = {}
        for b in parameters:
            plt.plot(ARP.trendLine(tn[b][0],100),label = "{}{}".format(a,b))
            tq[b] = [sum(tn[b][0][len(tn[b][0])//2:])/(len(tn[b][0])//2),ARP.stdDev(tn[b][0])]
        quantl.append(tq)
    
    print("Quant last-half comparisons:")
    for a in range(len(quantl)):
        #For each file:
        print(files[a])
        for b in quantl[a]:
            #For each data type:
            print("\t{}: Avg {}, StDev {}".format(b,quantl[a][b][0],quantl[a][b][1]))
        print()
    
    plt.legend()
    plt.show()

def groupGraph(prefix,loops,layer,param):
    
    t1,t2 = ARP.loadGroups(prefix,loops)
    for a in t1[layer]:
        plt.plot(ARP.trendLine(t1[layer][a][param],100),label = a)
        quant = t1[layer][a][param]
        quant = quant[len(quant)//2:]
        print("Last-half quantitative for L{}V{}: Start {}, Max {}, Min {}, Avg {}, StDev {}".format(layer,a,quant[0],max(quant),min(quant),sum(quant)/len(quant),ARP.stdDev(quant)))
    
    plt.legend()
    plt.show()
    
    stacklist = []
    #We know we have 3,5
    #Which means we know we're looking for t2[0][0-2][0-4]
    stacksz = 1
    for b in range(1,len(loops)): stacksz = stacksz * loops[b]
    lh = len(t2[0][0][0]["LFreqD"][0])//2 #looptype 0, val 0, index 0, data lfreqd, batches
    totlist = []
    for a in t2[0]:
        for b in t2[0][a]:
            totlist.append(b)
    '''   
    for a in range(len(totlist)):
        stackline = [a]
        for b in ["DiscE","GenE","LFreqD","Markov1","Markov2"]:
            sublist = totlist[a][b][0][lh:]
            stackline.append(sum(sublist)/len(sublist))
        stacklist.append(stackline)
    
    for a in range(5):
        stacklist.sort(key = lambda b:b[a+1])
        print("By parameter {}".format(["DiscE","GenE","LFreqD","Markov1","Markov2"][a]))
        for b in range(len(stacklist)):
            print("\t",stacklist[b][0],stacklist[b][a+1])
        
        print()'''

def collapsetest(filename):
    #How do we test for mode collapse, exactly? Is there a point at which it becomes irrecoverable?
    #How many unique patterns are there in the space of all unique words?
    wl = ARP.loadDump(filename)
    setl = [list(set(a)) for a in wl]
    
    patterns = []
    for a in setl:
        found = False
        for b in patterns:
            if all(c in b for c in a) and len(a) == len(b):
                found = True
        if not found:
            patterns.append(a)
    
    patsq = []
    for a in setl:
        for b in range(len(patterns)):
            if all(c in patterns[b] for c in a) and len(a) == len(patterns[b]):
                patsq.append(b)
    
    return [setl,patterns,patsq,wl]
    

def run(words,gibberish,out):
    #torch.manual_seed(0)
    wl = DataLoader(words,batch_size = 512,shuffle = True)
    
    gh = 3
    dh = 5
    
    gen_params = [
        [words.dl] + [150] * gh + [words.dl],
        [[" "," ","e"," "],[" "," ","e"," "]] + [[" "," ","e", " "]] * (gh - 2) + [[" "," ","t"," "]],
        "Adamax",
        [0.002, (0.6,0.99), 0]
        ]
    
    disc_params = [
        [words.dl] + [400] * dh + [1],
        [[" "," ","r"," "],[" "," ","r"," "]] + [[" "," ","r"," "]] * (dh - 2) + [[" "," "," "," "]],
        "Adamax",
        [0.01, (0.6,0.99), 0]
        ]
    
    testNet = GANModel(gen_params,disc_params,["WGAN","GP3"])
    testNet.Train(wl,3000,[words,gibberish],ratio = (6,1),run_name = "AROut/{}".format(out),eval_mode = True)
    testNet.saveGAN(out)
    
def parameterSearch(words,gibberish):
    #Loop through several sets of parameters
    wl = DataLoader(words,batch_size = 265,shuffle = True)
    
    current_index = 0
    
    for a in range(4):
        for b in range(4):
            torch.manual_seed(0)
            gh = 3
            nlg = ["r","e","l0.1","l0.2"][1]
            
            dh = 5
            nld = ["r","e","l0.1","l0.2"][0]
            
            #gd = "d0.{}".format(a*25)
            gd = "d0.3"
            
            gen_params = [
                [words.dl] + [200] * (gh) + [words.dl],
                [[" "," ",nlg," "],[" ",gd,nlg," "]] + [[" "," ",nlg," "]]*(gh-2) + [[" "," ","t"," "]],
                "Adamax",
                [((2*a)+3)/1000, (0.6,0.99), 0]
                ]
            
            dd = "d0.5"
            
            disc_params = [
                [words.dl] + [300] * (dh) + [1],
                [[" "," ",nld," "],[" ",dd,nld," "]] + [[" ",dd,nld," "]] * (dh-2) + [[" "," "," "," "]],
                "Adamax",
                [((2*b)+3)/1000, (0.6,0.99), 0]
                ]
            testNet = GANModel(gen_params,disc_params,["WGAN","GP10"])
            testNet.Train(wl,200,[words,gibberish],ratio = (5,1),run_name = "041824 LR Test {}.txt".format(current_index),eval_mode = True)
            current_index += 1
    
def classTest(words,eval_words,out_f):
    wl = DataLoader(words,batch_size = 512,shuffle = True)
    #el = DataLoader(eval_words,batch_size = 200,shuffle = True)
    cd = 6
    
    class_params = [
        [words.dl] + [300] * cd + [1],
        [["b","d0.9","e"," "]] + [["b","d0.9","e"," "]] * (cd - 1) + [[" "," ","t"," "]],
        "Adam",
        [0.015,(0.9,0.99),0]
        ]
    
    testClass = Classifier(class_params,["BCE","rs"])
    testClass.Train(wl,200,out_f,eval_data = eval_words,eval_mode = True)

def plotClass(in_f):
    crt = ARP.loadNewDump(in_f)
    plt.plot(ARP.trendLine(crt["T_Acc"][0],100),label = "T_Acc")
    plt.plot(ARP.trendLine(crt["V_Acc"][0],100),label = "V_Acc")
    plt.plot(ARP.trendLine(crt["Err"][0],100),label = "Err")
    plt.legend()
    plt.show()

def main():
    words = ARD.Wordset(["ARData/5_7_la.json"])
    words2 = ARD.Wordset(["ARData/5_7_pokel.json"])
    gibberish = ARD.Wordset(["ARData/5_7_l_g.json"])
    
    
    trainwords = ARD.Wordset(["ARData/031424 Pref a.json","ARData/031424 Pref b.json","ARData/031424 Pref c.json","ARData/031424 Pref d.json"],size_1 = True,smooth = True)
    evalwords = ARD.Wordset(["ARData/031424 Pref e.json"],size_1 = True)
    
    #classTest(trainwords,evalwords,"AROut/040924 classifier 1.txt")
    #plotClass("AROut/040924 classifier 1.txt")
    
    #run(words,gibberish,"050724 test 4")
    
    #loadtest = GanFromFile("032824 0")
    #quickGraph("AROut/043024 test 2.txt",ref = words.big_set)
    quickCompare(["AROut/050724 test 1.txt","AROut/050724 test 2.txt","AROut/050724 test 3.txt","AROut/050724 test 4.txt"],["Markov1"])
    
    #parameterSearch(words,gibberish)
    #groupGraph("AROut/041824 LR Test", [4,4],1,"Markov2")
        
        
    
    
main()