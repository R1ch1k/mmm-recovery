# When to trust your MMM

*One page, no equations. For the person who has to decide what to do with the model's answer.*

---

## The short version: put a guardrail on it

We built a fake world where we knew, exactly, how much each marketing channel really caused. Then
we gave a Marketing Mix Model everything it could reasonably ask for — ten years of clean weekly
data, no hidden factors, no channels moving in lockstep, and the exactly right shape of response
curve built in — and asked it to recommend a budget. We did this two hundred times, with two
hundred different random draws of the same clean world.

**Everything depends on how much freedom you give the optimiser.** This was the single most
effective thing we tested, and it is the one thing on this page you can do on Monday.

| | Let the model move budget freely | Cap every change at ±30%, never switch a channel off |
|---|---|---|
| Typical effect on sales vs. **leaving the budget alone** | **−1.58%** | **+0.30%** |
| Typical shortfall vs. the **best allocation available** | 2.73% | 0.32% |
| Runs where the advice was **worse than doing nothing** | **80%** (160 of 200) | **24%** (48 of 200) |
| Runs where it beat doing nothing | 20% | 76% |

Those are two different questions and the numbers are not interchangeable. The first row is the one
that matters on Monday: it compares acting on the model against not acting. The second compares it
against a perfect answer nobody can actually know.

Read the first column carefully. Not "less good than it could have been" — *worse than doing
nothing*, four times out of five, from a model given ideal data. The guardrail turns that around.

**But it is not a fix, and the reason is the uncomfortable part.** Once you apply realistic planning
rules, only **0.62% of total sales** is available from reallocating the budget at all. The prize is
that small. So the downside of acting on a badly identified model is larger than the upside of
acting on a well identified one, and the guardrail works by capping the downside — not by making
the model right.

We also set a standard before running any of this: the advice had to beat doing nothing 9 times in
10. Free, it managed 2 in 10. Guardrailed, 7.6 in 10. **Both fail.** The guardrail is damage
limitation, not a licence to trust the number.

Then we ran Google's own Meridian on the same data, on its default settings. It got the channel
contributions wrong by about the same margin.

## What this is not

**This is not "MMM is useless."** It is a specific finding about a specific thing: how much of your
sales a channel *caused*, and therefore where the next pound should go. That is a harder question
than the one MMM usually gets credit for answering.

**This is also not new.** Researchers at Wharton and London Business School published the core
identification problem in 2024, and Google papers flagged parts of it in 2017. What we added is the
price tag: we followed the error all the way through to the budget decision and measured what it
costs.

## Why it happens

Think of your spend data as a set of dots and the model's job as drawing a curve through them.

The curve has two things to decide at once: **how big** the effect is, and **how quickly it
saturates** — the point where another pound stops doing much. If your spending has always sat in
roughly the same range, an enormous effect that saturates early and a modest effect that saturates
late look *identical* over the range you actually spent in. The data cannot tell them apart.

We measured this directly. Holding four channels at their true settings and varying only TV's, out
of 780 candidate settings **177 of them fitted the data essentially as well as the true one** — and
across those near-identical fits, the estimated contribution of TV ranged from about **£44m to
£241m**. A five-and-a-half-fold spread in the answer, with nothing in the data to choose between
them.

The model still picks one. It has to. It reports it with a confidence interval. The interval is too
narrow, because it prices the wrong uncertainty.

**And the two answers imply different budgets.** That is the whole problem. Two models that agree
perfectly about last year disagree completely about what to do next year.

## The thing that makes this worse, not better

We checked whether the model was simply not searching hard enough. It was not.

When we started the search *at the exactly correct answer*, it walked away from it — finding a
better statistical fit while getting the actual contributions dramatically more wrong. When we gave
the budget optimiser more attempts, its recommendations got worse, not better.

**A better fit is not a better answer.** If your agency tells you the model has been improved
because the fit statistics went up, that is not evidence of anything.

## Would more spend variation fix it?

This is the useful question, so we tested it. We re-ran everything with weekly spend swinging
progressively harder — from very steady, through our baseline, up to a level of week-to-week jitter
no planner would ever deliberately introduce.

It helps. It does not fix it. At the most extreme setting — where a single channel's weekly spend
spans a factor of about thirteen between its quiet weeks and its heavy ones, and more than four
weeks in ten sit below half that channel's own average — the model's advice was **still worse than
doing nothing in 54% of runs**. Only the *ranking* of channels became reliable.

## Does flighting help? Yes — for the estimate, not for the decision

We then tested the thing a media planner would ask about immediately. Instead of spend merely
jittering, we made TV, video and out-of-home run the way they are actually bought: bursts of two to
six weeks live, dark weeks between, same annual budget concentrated into fewer weeks. Search and
social stayed always-on.

**It helped a lot — and only with the half that doesn't pay the bills.**

- The typical channel's contribution error nearly halved.
- Five and a half times as many runs got the contributions right.
- Channel *ranking* became reliable — the only time in this whole study that happened under
  something a media team would actually choose to do.
- And the budget recommendation showed **no measurable improvement**: it beat doing nothing 20.7% of
  the time, against 20.0% without flighting — a difference of one percentage point, well inside what
  two hundred runs can distinguish from chance.

So if your brand channels go dark between bursts, your model's numbers really are better than they
would be otherwise. That is a genuine finding and worth having.

**But you can learn which channels are better without learning how much better, and ranking without
sizing does not allocate a budget.** That gap is the single most important thing on this page.

## The conclusion nobody wants

**You cannot observe your way out of this. You have to experiment.** Deliberately turning spend up
and down — a holdout, a geo test, a bump-up test — creates the contrast that passively collected
history never will. The academic work reaches the same conclusion from the other direction: their
fix is a deliberate high-low spending pattern run for a period or two, not more history.

## What to do on Monday

**Questions worth asking about a model you're being shown:**

1. **How much has our spend actually varied?** If every channel has sat within ±20% of its average
   for three years, the model cannot know its saturation point. It will still print one.
2. **What would the answer be under a different but equally good-fitting setting?** If nobody can
   tell you, the uncertainty you've been shown is understated.
3. **Has any part of this been checked against an experiment?** A single geo holdout is worth more
   than a year of extra history.
4. **Is the recommendation asking for a big move?** The bigger the reallocation, the further it
   relies on the part of the curve you have never actually visited.

**What to do with the output:**

- **Put a guardrail on it.** Cap any single reallocation at ±30% and never let a model switch a
  channel off. This was the single most effective thing we tested: it cut worse-than-doing-nothing
  outcomes from 80% of runs to 24%, and turned a typical 1.58% loss of sales against leaving the
  budget alone into a 0.30% gain.
- **Use it for ranking, cautiously — to choose what to test, not what to fund.** Channel order was
  the only thing that became reliable under a plan a media team would actually choose, and even then
  it only just reached the bar. Rough ordering is a defensible way to pick the next experiment. It
  is not a basis for sizing a reallocation, which is the thing that failed.
- **Do not use it for large reallocations.** That is precisely the use that failed.
- **Treat "incremental" claims as attribution unless an experiment says otherwise.** That is the
  point of the whole exercise: attributable is not the same as incremental.
- **Move in small steps and measure.** A modest reallocation you can actually verify beats an
  optimal-looking one you cannot.

## Where this might not apply

We tested one kind of simulated world. A business with genuinely different dynamics might be easier
to measure. Our model is a good one but not every MMM — a different tool could do better, though the
one production tool we tested did not. And if your team already runs experiments and feeds the
results back into the model, most of this does not describe you. That is the point: **it is the
experiments that make the model trustworthy, not the model that makes the experiments unnecessary.**

---

*Full method, numbers and caveats: [`../README.md`](../README.md) and
[`../PREREGISTRATION.md`](../PREREGISTRATION.md). The thresholds were written down and committed
before any of this was run.*
