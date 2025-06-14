# POMO: Policy Optimization with Multiple Optima for Reinforcement Learning

Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, Seungjai Min Samsung SDS {y.d.kwon, jinho12.choo, bjip.kim, iljoo.yoon, gyj.gwon, seungjai.min}@samsung.com

# Abstract

In neural combinatorial optimization (CO), reinforcement learning (RL) can turn a deep neural net into a fast, powerful heuristic solver of NP-hard problems. This approach has a great potential in practical applications because it allows nearoptimal solutions to be found without expert guides armed with substantial domain knowledge. We introduce Policy Optimization with Multiple Optima (POMO), an end-to-end approach for building such a heuristic solver. POMO is applicable to a wide range of CO problems. It is designed to exploit the symmetries in the representation of a CO solution. POMO uses a modified REINFORCE algorithm that forces diverse rollouts towards all optimal solutions. Empirically, the lowvariance baseline of POMO makes RL training fast and stable, and it is more resistant to local minima compared to previous approaches. We also introduce a new augmentation-based inference method, which accompanies POMO nicely. We demonstrate the effectiveness of POMO by solving three popular NP-hard problems, namely, traveling salesman (TSP), capacitated vehicle routing (CVRP), and 0-1 knapsack (KP). For all three, our solver based on POMO shows a significant improvement in performance over all recent learned heuristics. In particular, we achieve the optimality gap of $0 . 1 4 \%$ with TSP100 while reducing inference time by more than an order of magnitude.

# 1 Introduction

Combinatorial optimization (CO) is an important problem in logistics, manufacturing and distribution supply chain, and sequential resource allocation. The problem is studied extensively by Operations Research (OR) community, but the real-world CO problems are ubiquitous, and each problem is different from one another with its unique constrains. Moreover, these constrains tend to vary rapidly with a changing work environment. Devising a powerful and efficient algorithm that can be applied uniformly under various conditions is tricky, if not impossible. Therefore, many CO problems faced in industries have been commonly dealt with hand-crafted heuristics, despite their drawbacks, engineered by local experts.

In the field of computer vision (CV) and natural language processing (NLP), classical methods based on manual feature engineering by experts have now been superseded by automated end-to-end deep learning algorithms [1, 2, 3, 4, 5]. Tremendous progresses in supervised learning, where a mapping from training inputs to their labels is learned, has made this remarkable transition possible. Unfortunately, supervised learning is largely unfit for most CO problems because one cannot have an instant access to optimal labels. Rather, one should make use of the scores, that are easily calculable for most CO solutions, to train a model. Reinforcement learning paradigm suits combinatorial optimization problems very well.

Recent approaches in deep reinforcement learning (RL) have been promising [6], finding close-tooptimal solutions to the abstracted NP-hard CO problems such as traveling salesman (TSP) [7, 8, 9, 10,

11, 12], capacitated vehicle routing (CVRP) [10, 11, 13, 14, 15], and 0-1 knapsack (KP) [7] in superior speed. We contribute to this line of group effort in the deep learning community by introducing Policy Optimization with Multiple Optima (POMO). POMO offers a simple and straightforward framework that can automatically generate a decent solver. It can be applied to a wide range of general CO problems because it uses symmetry in the CO itself, found in sequential representation of CO solutions.

We demonstrate the effectiveness of POMO by solving three NP-hard problems aforementioned, namely TSP, CVRP, and KP, using the same neural net and the same training method. Our approach is purely data-driven, and the human guidance in the design of the training procedure is kept to minimal. More specifically, it does not require problem-specific hand-crafted heuristics to be inserted into its algorithms. Despite its simplicity, our experiments confirm that POMO achieves superior performances in reducing the optimality gap and inference time against all contemporary neural RL approaches.

The contribution of this paper is three-fold:

• We identify symmetries in RL methods for solving CO problems that lead to multiple optima. Such symmetries can be leveraged during neural net training via parallel multiple rollouts, each trajectory having a different optimal solution as its goal for exploration. • We devise a new low-variance baseline for policy gradient. Because this baseline is derived from a group of heterogeneous trajectories, learning becomes less vulnerable to local minima. • We present the inference method based on multiple greedy rollouts that is more effective than the conventional sampling inference method. We also introduce an instance augmentation technique that can further exploit symmetries of CO problems at the inference stage.

# 2 Related work

Deep RL construction methods. Bello et al. [7] use a Pointer Network (PtrNet) [16] in their neural combinatorial optimization framework. As one of the earliest deep RL approaches, they have employed the actor-critic algorithm [17] and demonstrated neural combinatorial optimization that achieves close-to-optimal results in TSP and KP. The PtrNet model is based on the sequence-tosequence architecture [3] and uses attention mechanism [18]. Narari et al. [19] have further improved the PtrNet model.

Differentiated from the previous recurrent neural network (RNN) based approaches, Attention Model [10] opts for the Transformer [4] architecture. REINFORCE [20] with a greedy rollout baseline trains Attention Model, similar to self-critical training [21]. Attention Model has been applied to routing problems including TSP, orienteering (OP), and VRP. Peng et al. [22] show that a dynamic use of Attention Model can enhance its performance.

Dai et al. propose Struct2Vec [23]. Using Struct2Vec, Khalil et al. [8] have developed a deep Qlearning [24] method to solve TSP, minimum vertex cut and maximum cut problems. Partial solutions are embedded as graphs, and the deep neural net estimates the value of each graph.

Inference techniques. Once the neural net is fully trained, inference techniques can be used to improve the quality of solutions it produces. Active search [7] optimizes the policy on a single test instance. Sampling method [7, 10] is used to choose the best among the multiple solution candidates. Beam search [16, 9] uses advanced strategies to improve the efficiency of sampling. Classical heuristic operations as post-processing may also be applied on the solutions produced by the neural net [25, 22] to further enhance their quality.

Deep RL improvement methods. POMO belongs to the category of construction type RL method summarized above, in which a CO solution is created by the neural net in one shot. There is, however, another important class of RL approach for solving CO problems that combines machine learning with the existing heuristic methods. A neural net can be trained to guide local search algorithm, which iteratively finds a better solution based on the previous ones until the time budget runs out. Such improvement type RL methods have been demonstrated with outstanding results by many, including Wu et al. [11] and Costa et al. [12] for TSP and Chen & Tian [13], Hottung & Tierney [14],

![](images/6f4ce0e5e99cf5f6ec3ad08f516c6379ecb96041f10212ac296a8c6fbd9441d8.jpg)  
Figure 1: Multiple optimal solutions of TSP highlighted in tree search. For the given instance of 5-node TSP problem, there exists only one unique optimal solution (LEFT). But when this solution is represented as a sequence of nodes, multiple representations exist (RIGHT).

Lu et al. [15] for CVRP. We note that formulation of improvement heuristics on top of POMO should be possible and can be an important further research topic.

# 3 Motivation

Assume we are given a combinatorial optimization problem with a group of nodes $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { M } \}$ and have a trainable policy net parameterized by $\theta$ that can produce a valid solution to the problem. A solution $\pmb { \tau } = ( a _ { 1 } , \dots , a _ { M } )$ , where the $i$ th action $a _ { i }$ can choose a node $v _ { j }$ , is generated by the neural net autoregressively one node at a time, following the stochastic policy

$$
\pi _ { t } = { \left\{ \begin{array} { l l } { p _ { \theta } ( a _ { t } \mid s ) } & { { \mathrm { f o r } } t = 1 } \\ { p _ { \theta } ( a _ { t } \mid s , a _ { 1 : t - 1 } ) } & { { \mathrm { f o r } } t \in \{ 2 , 3 , \ldots , M \} } \end{array} \right. }
$$

where $s$ is the state defined by the problem instance.

In many cases, a solution of a CO problem can take multiple forms when represented as a sequence of nodes. A routing problem that contains a loop, or a CO problem finding a “set” of items have such characteristics, to name a few. Take TSP for an example: if $\pmb { \tau } = ( v _ { 1 } , v _ { 2 } , v _ { 3 } , v _ { 4 } , v _ { 5 } )$ is an optimal solution of a 5-node TSP, then $\pmb { \tau } ^ { \prime } = ( v _ { 2 } , v _ { 3 } , v _ { 4 } , v _ { 5 } , v _ { 1 } )$ also represents the same optimal solution (Figure 1).

When asked to produce the best possible answer within its capability, a perfectly logical agent with prior knowledge of equivalence among such sequences should produce the same solution regardless of which node it chooses to output first. This, however, has not been the case in the previous learningbased models. As is clear in Equation (1), the starting action $( a _ { 1 } )$ heavily influences the rest of the agent’s course of actions $( a _ { 2 } , a _ { 3 } , \dotsc , a _ { M } )$ , when in fact any choice for $a _ { 1 }$ should be equally good1. We seek to find a policy optimization method that can fully exploit this symmetry.

# 4 Policy Optimization with Multiple Optima (POMO)

# 4.1 Explorations from multiple starting nodes

POMO begins with designating $N$ different nodes $\{ a _ { 1 } ^ { 1 } , a _ { 1 } ^ { 2 } , \ldots , a _ { 1 } ^ { N } \}$ as starting points for exploration (Figure 2, (b)). The network samples $N$ solution trajectories $\{ \pmb { \tau } ^ { 1 } , \pmb { \tau } ^ { 2 } , \dots , \pmb { \tau } ^ { N } \}$ via Monte-Carlo method, where each trajectory is defined as a sequence

$$
\begin{array} { r } { \pmb { \tau } ^ { i } = ( a _ { 1 } ^ { i } , a _ { 2 } ^ { i } , \dotsc , a _ { M } ^ { i } ) \qquad \mathrm { f o r } \ i = 1 , 2 , \dotsc , N . } \end{array}
$$

In previous work that uses RNN- or Transformer-style neural architectures, the first node from multiple sample trajectories is always chosen by the network. A trainable START token, a legacy from NLP where those models originate, is fed into the network, and the first node is returned (Figure 2, (a)). Normally, the use of such START token is sensible, because it allows the machine to learn to find the “correct” first move that leads to the best solution. In the presence of multiple “correct” first moves, however, it forces the machine to favor particular starting points, which may lead to a biased strategy.

![](images/b5f968bbbfdbf9b8ff20e48917d5e0daff8fecf0683212abaae1ac94abe9762d.jpg)  
Figure 2: (a) Common method for generating a single solution trajectory $( \tau )$ based on START token scheme. (b) POMO method for multiple trajectory $\{ \pmb { \tau } ^ { 1 } , \pmb { \tau } ^ { 2 } , . . . , \overset { \smile } { \boldsymbol { \tau } } ^ { N } \}$ generation in parallel with a different starting node for each trajectory.

When all first moves are equally good, therefore, it is wise to apply entropy maximization techniques [27] to improve exploration. Entropy maximization is typically carried out by adding an entropy regularization term to the objective function of RL. POMO, however, directly maximizes the entropy on the first action by forcing the network to always produce multiple trajectories, all of them contributing equally during training.

Note that these trajectories are fundamentally different from repeatedly sampled $N$ trajectories under the START token scheme [28]. Each trajectory originating from a START token stays close to a single optimal path, but $N$ solution trajectories of POMO will closely match $N$ different node-sequence representations of the optimal solution. Conceptually, explorations by POMO are analogous to guiding a student to solve the same problem repeatedly from many different angles, exposing her to a variety of problem-solving techniques that would otherwise be unused.

# 4.2 A shared baseline for policy gradients

POMO is based on the REINFORCE algorithm [20]. Once we sample a set of solution trajectories $\{ \pmb { \tau } ^ { 1 } , \pmb { \tau } ^ { 2 } , \dots , \pmb { \tau } ^ { N } \}$ , we can calculate the return (or total reward) $\bar { R } ( \pmb { \tau } ^ { i } )$ of each solution $\boldsymbol { \tau ^ { i } }$ . To maximize the expected return $J$ , we use gradient ascent with an approximation

$$
\nabla _ { \theta } J ( \theta ) \approx \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( R ( \boldsymbol { \tau ^ { i } } ) - b ^ { i } ( \boldsymbol { s } ) ) \nabla _ { \theta } \log p _ { \theta } ( \boldsymbol { \tau ^ { i } } | \boldsymbol { s } )
$$

where $\begin{array} { r } { p _ { \theta } ( \pmb { \tau } ^ { i } | s ) \equiv \prod _ { t = 2 } ^ { M } p _ { \theta } ( a _ { t } ^ { i } | s , a _ { 1 : t - 1 } ^ { i } ) } \end{array}$ .

In Equation (3), $b ^ { i } ( s )$ is a baseline that one has some freedom of choice to reduce the variance of the sampled gradients. In principle, it can be a function of $a _ { 1 } ^ { i }$ , assigned differently for each trajectory $\boldsymbol { \tau } ^ { i }$ . In POMO, however, we use the shared baseline,

$$
b ^ { i } ( s ) = b _ { \mathrm { s h a r e d } } ( s ) = { \frac { 1 } { N } } \sum _ { j = 1 } ^ { N } R ( \pmb { \tau } ^ { j } ) \quad \mathrm { f o r ~ a l l } i .
$$

Algorithm 1 presents the POMO training with mini-batch.

POMO baseline induces less variance in the policy gradients compared to the greedy-rollout baseline [10]. The advantage term in Equation (3), $R \dot { ( } \pmb { \tau } ^ { i } ) - b ^ { i } ( s )$ , has zero mean for POMO, whereas the greedy-rollout baseline results in negative advantages most of the time. This is because samplerollouts (following softmax of the policy) have difficulty in surpassing greedy-rollouts (following argmax of the policy) in terms of the solution qualities, as will be demonstrated later in this paper. Also, as an added benefit, POMO baseline can be computed efficiently compared to other baselines used in previous deep-RL construction methods, which require forward passes through either a separately trained network (Critic [7, 8]) or the cloned policy network (greedy-rollout [10]).

Most importantly, however, the shared baseline used by POMO makes RL training highly resistant to local minima. After generating $N$ solution trajectories $\{ \pmb { \tau } ^ { 1 } , \pmb { \tau } ^ { 2 } , \dots , \pmb { \tau } ^ { N } \}$ , if we do not use the shared

# Algorithm 1 POMO Training

<html><body><table><tr><td colspan="2">1: procedure TRAINING(training set S, number of starting nodes per sample N, number of training</td></tr><tr><td>steps T,batch size B)</td><td>initialize policy network parameter θ</td></tr><tr><td>2:</td><td></td></tr><tr><td>3:</td><td>for step= 1,...,T do</td></tr><tr><td>4:</td><td>S𝑖 ←SAMPLEINPUT(S)∀i∈{1,...,B} {α¹,α²,...,αN}← SELECTSTARTNODES(si）∀i∈{1,...,B}</td></tr><tr><td>5:</td><td></td></tr><tr><td>6:</td><td>T ← SAMPLEROLLOUT(α²,Si,θ) ∀i∈{1,...,B},∀j∈{1,...,N}</td></tr><tr><td>7:</td><td>bi←∑_B(t)） Vie{1,n,B)</td></tr><tr><td>8:</td><td>V0J（θ)←BN∑1∑1(R(t）-bi）Vlogpe(t}） 0←0+αVθJ(0)</td></tr><tr><td colspan="2">9: 10:</td></tr><tr><td colspan="2">end for</td></tr><tr><td colspan="2">11:end procedure</td></tr></table></body></html>

baseline but strictly stick to the greedy-rollout baseline scheme [10] instead, each sample-rollout $\boldsymbol { \tau ^ { i } }$ would be assessed independently. Actions that produced $\boldsymbol { \tau ^ { i } }$ would be reinforced solely by how much better (or worse) it performed compared to its greedy-rollout counterpart with the same starting node $a _ { 1 } ^ { i }$ . Because this training method is guided by the difference between the two rollouts produced by two closely-related networks, it is likely to converge prematurely at a state where both the actor and the critic underperform in a similar fashion. With the shared baseline, however, each trajectory now competes with $N - 1$ other trajectories where no two trajectories can be identical. With the increased number of heterogeneous trajectories all contributing to setting the baseline at the right level, premature converge to a suboptimal policy is heavily discouraged.

# 4.3 Multiple greedy trajectories for inference

Construction type neural net models for CO problems have two modes for inference in general. In “greedy mode,” a single deterministic trajectory is drawn using argmax on the policy. In “sampling mode,” multiple trajectories are sampled from the network following the probabilistic policy. Sampled solutions may return smaller rewards than the greedy one on average, but sampling can be repeated as many times as needed at the computational cost. With a large number of sampled solutions, some solutions with rewards greater than that of the greedy rollout can be found.

Using the multi-starting-node approach of POMO, however, one can produce not just one but multiple greedy trajectories. Starting from $N$ different nodes $\{ a _ { 1 } ^ { 1 } , a _ { 1 } ^ { 2 } , \ldots , a _ { 1 } ^ { N } \}$ , $N$ different greedy trajectories can be acquired deterministically, from which one can choose the best similarly to the “sampling mode” approach. $N$ greedy trajectories are in most cases superior than $N$ sampled trajectories.

Instance augmentation. One drawback of POMO’s multi-greedy inference method is that $N$ , the number of greedy rollouts one can utilize, cannot be arbitrarily large, as it is limited to a finite number of possible starting nodes. In certain types of CO problems, however, one can bypass this limit by introducing instance augmentation. It is a natural extension from the core idea of POMO, seeking different ways to arrive at the same optimal solution. What if you can reformulate the problem, so that the machine sees a different problem only to arrive at the exact same solution? For example, one can flip or rotate the coordinates of all the nodes in a 2D routing optimization problem and generate another instance, from which more greedy trajectories can be acquired.

Instance augmentation is inspired by self-supervised learning techniques that train neural nets to learn the equivalence between rotated images [29]. For CV tasks, there are conceptually similar test-time augmentation techniques such as “multi-crop evaluation” [30] that enhance neural nets’ performance at the inference stage. Applicability and multiplicative power of instance augmentation technique on CO tasks depend on the specifics of a problem and also on the policy network model that one uses. More ideas on instance augmentation are described in Appendix.

Algorithm 2 describes POMO’s inference method, including the instance augmentation.

# Algorithm 2 POMO Inference

<html><body><table><tr><td colspan="2">1: procedure INFERENCE(input s, policy Tθ, number of starting nodes N, number of transforms</td></tr><tr><td>K) 2:</td><td>{s1,S2,...,SK} ← AUGMENT(s)</td></tr><tr><td>3:</td><td></td></tr><tr><td>4:</td><td>{,Ω²,.,ON} ←SELECTSTARTNODES(sk）∀k∈{1,..,K}</td></tr><tr><td>5:</td><td>τ ← GREEDYROLLOUT(α,s,πθ）∀j ∈{1,..,N},∀k ∈ {1,.,K}</td></tr><tr><td></td><td>kmax,Jmax ←argmaxk,j R(Tκ)</td></tr><tr><td>6:</td><td>Jmax return Tkmax</td></tr><tr><td colspan="2">7: end procedure</td></tr></table></body></html>

# 5 Experiments

The Attention Model. All of our POMO experiments use the policy network named Attention Model (AM), introduced by Kool et al. [10]. The AM is particularly well-suited for POMO, although we emphasize that POMO is a general RL method, not tied to a specific structure of the policy network. The AM consists of two main components, an encoder and a decoder. The majority of computation happens inside the heavy, multi-layered encoder, through which information of each node and its relation with other nodes is embedded as a vector. The decoder then generates a solution sequence autoregressively using these vectors as the keys for its dot-product attention mechanism.

To apply POMO, we need to draw multiple $( N )$ trajectories for one instance of a CO problem. This does not affect the encoding procedure on the AM, as the encoding is required only once regardless of the number of trajectories one needs to generate. The decoder of the AM, on the other hand, needs to process $N$ times more computations for POMO. By stacking $N$ queries into a single matrix and passing it to the attention mechanism (a natural usage of attention), $N$ trajectories can be generated efficiently in parallel.

Problem setup. For TSP and CVRP, we solve the problems with a setup as prescribed in Kool et al. [10]. For 0-1 KP, we follow the setup by Bello et al. [7].

Training. For all experiments, policy gradients are averaged from a batch of 64 instances. Adam optimizer [31] is used with a learning rate $\eta \ : = \ : 1 0 ^ { - 4 }$ and a weight decay $\cdot \_ { L _ { 2 } }$ regularization) $\bar { w } = 1 0 ^ { - 6 }$ . To keep the training condition simple and identical for all experiments we have not applied a decaying learning rate, although we recommend a fine-tuned decaying learning rate in practice for faster convergence. We define one epoch 100,000 training instances generated randomly on the fly. Training time varies with the size of the problem, from a couple of hours to a week. In the case of TSP100, for example, one training epoch takes about 7 minutes on a single Titan RTX GPU. We have waited as long as 2,000 epochs $\sim 1$ week) to observe full converge, but as shown in Figure 3 (b), most of the learning is already completed by 200 epochs ( $\sim 1$ day).

Inference. We follow the convention and report the time for solving 10,000 random instances of each problem. For routing problems, we have performed inferences with and without $\times 8$ instance augmentation, using the coordinate transformations listed in Table 1. No instance augmentation is used for KP because there is no straightforward way to do so.

Table 1: Unit square transformations   

<html><body><table><tr><td>f(x,y)</td></tr><tr><td>(x,y) (y,x) (x,1-y) (y,1-x)</td></tr><tr><td>(1-x,y) (1-y,x)</td></tr><tr><td>(1-x,1-y) (1-y,1-x)</td></tr></table></body></html>

Originally, Kool et al. [10] have trained the AM using REINFORCE with a greedy rollout baseline. It is interesting to see how much the performance improves when POMO is used for training instead. For a concrete ablation study, however, the two separately trained neural nets must be evaluated in the same way. Because POMO inference method chooses the best from multiple answers, even without the instance augmentation, this gives POMO an unfair advantage. Therefore, we have additionally performed the inference in “single-trajectory mode” on our POMO-trained network, in which a random starting node is chosen to draw a single greedy rollout.

Note that averaged inference results can fluctuate when they are based on a small (10,000) test set of random instances. In order to avoid confusion for the readers, we have slightly modified some of the averaged path lengths of TSP results in Table 2 (based on the reported optimality gaps) so that they are consistent with the optimal values we computed (using more than 100,000 samplings), 3.83 and 5.69 for TSP20 and TSP50, respectively. For CVRP and KP, there are even larger sampling errors than TSP, and thus we are more careful in the presentation of the results in this case. We display “gaps” in the result tables only when they are based on the same test sets.

![](images/5ab3046e6a1d76d1b79312fdf84422f2cec855c18368b94ccc56a650f8aa9e90.jpg)  
Figure 3: Learning curves for TSP50 and TSP100 of REINFORCE with a greedy rollout baseline [10] (blue) and those of POMO (orange, green, and red) made by three different inference methods on the same neural net (AM). After each training epoch, we generate 10,000 random instances on the fly and use them as a validation set.

Code. Our implementation of POMO on the AM using PyTorch is publicly available2. We also share a trained model for each problem and its evaluation code.

# 5.1 Traveling salesman problem

We implement POMO by assigning every node to be a starting point for a sample rollout for TSP.   
That is, the number of starting nodes $( N )$ we use is 20 for TSP20, 50 for TSP50, and 100 for TSP100.

In Table 2 we compare the performance of POMO on TSP with other baselines. The first group of baselines shown at the top are results from Concorde [32] and a few other representative non-learningbased heuristics. We have run Concorde ourselves to get the optimal solutions, and other solvers’ data are adopted from Wu et al. [11] and Kool et al. [10]. The second group of baselines are from deep RL improvement-type approaches found in the literature [9, 11, 12]. In the third group, we present the results from the AM that is trained by our implementation of REINFORCE with a greedy rollout baseline [10] instead of POMO.

Table 2: Experiment results on TSP   

<html><body><table><tr><td>Method</td><td>Len.</td><td>TSP20 Gap</td><td>Time</td><td>Len.</td><td>TSP50 Gap</td><td>Time</td><td>Len.</td><td>TSP100 Gap</td><td>Time</td></tr><tr><td>Concorde</td><td>3.83</td><td></td><td>(5m)</td><td>5.69</td><td></td><td>(13m)</td><td>7.76</td><td></td><td>(1h)</td></tr><tr><td>LKH3</td><td>3.83</td><td>0.00%</td><td>(42s)</td><td>5.69</td><td>0.00%</td><td>(6m)</td><td>7.76</td><td>0.00%</td><td>(25m)</td></tr><tr><td>Gurobi</td><td>3.83</td><td>0.00%</td><td>(7s)</td><td>5.69</td><td>0.00%</td><td>(2m)</td><td>7.76</td><td>0.00%</td><td>(17m)</td></tr><tr><td>OR Tools</td><td>3.86</td><td>0.94%</td><td>(1m)</td><td>5.85</td><td>2.87%</td><td>(5m)</td><td>8.06</td><td>3.86%</td><td>(23m)</td></tr><tr><td>Farthest Insertion</td><td>3.92</td><td>2.36%</td><td>(1s)</td><td>6.00</td><td>5.53%</td><td>(2s)</td><td>8.35</td><td>7.59%</td><td>(7s)</td></tr><tr><td>GCN [9], beam search</td><td>3.83</td><td>0.01%</td><td>(12m)</td><td>5.69</td><td>0.01%</td><td>(18m)</td><td>7.87</td><td>1.39%</td><td>(40m)</td></tr><tr><td>Improv.[11], {5000}</td><td>3.83</td><td>0.00%</td><td>(1h)</td><td>5.70</td><td>0.20%</td><td>(1h)</td><td>7.87</td><td>1.42%</td><td>(2h)</td></tr><tr><td>Improv.[12], {2000}</td><td>3.83</td><td>0.00%</td><td>(15m)</td><td>5.70</td><td>0.12%</td><td>(29m)</td><td>7.83</td><td>0.87%</td><td>(41m)</td></tr><tr><td>AM [10], greedy</td><td>3.84</td><td>0.19%</td><td>(<1s)</td><td>5.76</td><td>1.21%</td><td>(1s)</td><td>8.03</td><td>3.51%</td><td>(2s)</td></tr><tr><td>AM[10], sampling</td><td>3.83</td><td>0.07%</td><td>(1m)</td><td>5.71</td><td>0.39%</td><td>(5m)</td><td>7.92</td><td>1.98%</td><td>(22m)</td></tr><tr><td>POMO, single trajec.</td><td>3.83</td><td>0.12%</td><td>(<1s)</td><td>5.73</td><td>0.64%</td><td>(1s)</td><td>7.84</td><td>1.07%</td><td>(2s)</td></tr><tr><td>POMO, no augment.</td><td>3.83</td><td>0.04%</td><td>(<1s)</td><td>5.70</td><td>0.21%</td><td>(2s)</td><td>7.80</td><td>0.46%</td><td>(11s)</td></tr><tr><td>POMO,×8 augment.</td><td>3.83</td><td>0.00%</td><td>(3s)</td><td>5.69</td><td>0.03%</td><td>(16s)</td><td>7.77</td><td>0.14%</td><td>(1m)</td></tr></table></body></html>

Given 10,000 random instances of TSP20 and TSP50, POMO finds near-optimal solutions with optimality gaps of $0 . 0 0 0 6 \%$ in seconds and $0 . 0 2 5 \%$ in tens of seconds, respectively. For TSP100, POMO achieves the optimality gap of $0 . 1 4 \%$ in a minute, outperforming all other learning-based heuristics significantly, both in terms of the quality of the solutions and the time it takes to solve.

In the table, results under “AM, greedy” method and “POMO, single trajec.” method are both from the identical network structure that is tested by the same inference technique. The only difference was training, so the substantial improvement (e.g. from $3 . 5 1 \%$ to $1 . 0 7 \%$ in optimality gap on TSP100) indicates superiority of the POMO training method. As for the inference techniques, it is shown that the combined use of multiple greedy rollouts of POMO and the $\times 8$ instance augmentation can reduce the optimality gap even further, by an order of magnitude.

Learning curves of TSP50 and TSP100 in Figure 3 show that POMO training is more stable and sample-efficient. In reading these graphs, one should keep in mind that POMO uses $N$ -times more trajectories than simple REINFORCE for each training epoch. POMO training time is, however, comparable to that of REINFORCE, thanks to the parallel processing on trajectory generation. For example, TSP100 training takes about 7 minutes per epoch for POMO while it take 6 minutes for REINFORCE.

# 5.2 Capacitated vehicle routing problem

When POMO trains a policy net, ideally it should use only the “good” starting nodes from which one can roll out optimal solutions. But, unlike TSP, not all nodes in CVRP can be the first steps for optimal trajectories (see Figure 4) and there is no way of figuring out which of the nodes are good without actually knowing the optimal solution a priori. One way to resolve this issue is to add a secondary network that returns candidates for optimal starting nodes to be used by POMO. We leave this approach for future research, however, and in our CVRP experiment we stick to the same policy net that we have used for TSP without an upgrade. We simply use all nodes as starting nodes for POMO exploration regardless of whether they are good or bad.

This naive way of applying POMO can still make a powerful solver. Experiment results on CVRP with 20, 50, and

![](images/e928abd4c49b3d3bc03de447f3b6bbc81b0d11787fb92cf5af756eab41e07fab.jpg)  
Figure 4: An optimal solution of a 20- node CVRP plotted as a graph. For an agent that makes selections in the counter-clock-wise direction3, there are only three sequence representations of the optimal solution available: $\scriptstyle \tau ^ { 1 } , \tau ^ { 2 }$ , and $\hat { \mathbf { \tau } } ^ { 3 }$ .

100 customer nodes are reported in Table 3, and POMO is shown to outperform simple REINFORCE by a large margin. Note that there is no algorithm yet that can find optimal solutions of 10,000 random

Table 3: Experiment results on CVRP   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="3">CVRP20</td><td colspan="3">CVRP50</td><td colspan="3">CVRP100</td></tr><tr><td>Len.</td><td>Gap</td><td>Time</td><td>Len.</td><td>Gap</td><td>Time</td><td>Len.</td><td>Gap</td><td>Time</td></tr><tr><td>LKH3 OR Tools</td><td>6.12 6.42</td><td>4.84%</td><td>(2h) (2m)</td><td>10.38 11.22</td><td>8.12%</td><td>(7h) (12m)</td><td>15.68 17.14</td><td>9.34%</td><td>(12h) (1h)</td></tr><tr><td>NeuRewriter[13] NLNS [14]</td><td>6.16 6.19</td><td></td><td>(22m) (7m)</td><td>10.51 10.54</td><td></td><td>(18m) (24m)</td><td>16.10 15.99</td><td></td><td>(1h) (1h)</td></tr><tr><td>L2I [15] AM[10], greedy</td><td>6.12 6.40</td><td>4.45%</td><td>(12m) (<1s)</td><td>10.35 10.93</td><td>5.34%</td><td>(17m) (1s)</td><td>15.57 16.73</td><td>6.72%</td><td>(24m) (3s)</td></tr><tr><td>AM[10], sampling</td><td>6.24</td><td>1.97%</td><td>(3m)</td><td>10.59</td><td>2.11%</td><td>(7m)</td><td>16.16</td><td>3.09%</td><td>(30m)</td></tr><tr><td>POMO, single trajec.</td><td>6.35</td><td>3.72%</td><td>(<1s)</td><td>10.74</td><td>3.52%</td><td>(1s)</td><td>16.15</td><td>3.00%</td><td>(3s)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>POMO, no augment.</td><td>6.17</td><td>0.82%</td><td>(1s)</td><td>10.49</td><td>1.14%</td><td>(4s)</td><td>15.83</td><td>0.98%</td><td>(19s)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>POMO, ×8 augment.</td><td>6.14</td><td>0.21%</td><td>(5s)</td><td>10.42</td><td>0.45%</td><td>(26s)</td><td>15.73</td><td>0.32%</td><td>(2m)</td></tr></table></body></html>

3Empirically, we find that neural net-based agents choose nodes in an orderly fashion.

Table 4: Experiment results on KP   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">KP50</td><td colspan="2">KP100</td><td colspan="2">KP200</td></tr><tr><td>Score</td><td>Gap</td><td>Score</td><td>Gap</td><td>Score</td><td>Gap</td></tr><tr><td>Optimal</td><td>20.127</td><td>-</td><td>40.460</td><td></td><td>57.605</td><td></td></tr><tr><td>Greedy Heuristics Pointer Net [7], greedy</td><td>19.917 19.914</td><td>0.210</td><td>40.225</td><td>0.235 0.243</td><td>57.267</td><td>0.338</td></tr><tr><td>AM[10], greedy</td><td>19.957</td><td>0.213 0.173</td><td>40.217 40.249</td><td>0.211</td><td>57.271 57.280</td><td>0.334 0.325</td></tr><tr><td>POMO, single trajec.</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>19.997</td><td>0.130</td><td>40.335</td><td>0.125</td><td>57.345</td><td>0.260</td></tr><tr><td>POMO, no augment.</td><td>20.120</td><td>0.007</td><td>40.454</td><td>0.006</td><td>57.597</td><td>0.008</td></tr></table></body></html>

CVRP instances in a reasonable time, so the “Gap” values in the table are given relative to LKH3 [33] results. POMO has a smaller gap in CVRP100 $( 0 . 3 2 \% )$ than in CVRP50 $( 0 . 4 5 \% )$ , which is probably due to LKH3 falling faster in performance than POMO as the size of the problem grows.

In fact, L2I recently developed by Lu et al. [15] performs better than LKH3, making it the first deep RL-based CVRP solver that beats classical non-learning-based OR methods. Their achievement is a milestone in the application of the deep learning to OR. To emphasize the differences between POMO and L2I (other than the speed), POMO is a general RL tool that can be applied to many different CO problems in a purely data-driven way. One the other hand, L2I is a specialized routing problem solver based on a handcrafted pool of improvement operators. Because POMO is a construction method and L2I is an improvement type, it is possible to combine the two methods to produce even better results.

# 5.3 0-1 knapsack problem

We choose KP to demonstrate flexibility of POMO beyond routing problems. Similarly to the case of CVRP, we reuse the neural net for TSP and take the naive approach that uses all items given in the instance as the first steps for rollouts, avoiding an additional, more sophisticated “SelectStartNodes” network to be devised. In solving KP, a weight and a value of each item replaces x- and y-coordinate of each node of TSP. As the network generates a sequence of items, we put these items into the knapsack one by one until the bag is full, at which point we terminate the sequence generation.

In Table 4, the POMO results are compared with the optimal solutions based on dynamic programming, as well as those by greedy heuristics and our implementation of PtrNet [7] and the original AM method [10]. Even without the instance augmentation, POMO greatly improves the quality of the solutions one can acquire from a deep neural net.

# 6 Conclusion

POMO is a purely data-driven combinatorial optimization approach based on deep reinforcement learning, which avoids hand-crafted heuristics built by domain experts. POMO leverages the existence of multiple optimal solutions of a CO problem to efficiently guide itself towards the optimum, during both the training and the inference stages. We have empirically evaluated POMO with traveling salesman (TSP), capacitated vehicle routing (CVRP), and 0-1 knapsack (KP) problems. For all three problems, we find that POMO achieves the state-of-the-art performances in closing the optimality gap and reducing the inference time over other construction-type deep RL methods.

# Broader Impact

This work can facilitate the use of reinforcement learning approach based on deep neural net as a replacement for traditional heuristic methods used in operations of many sectors of business. Better and easier-to-use optimization tools will increase the productivity, but may lead to automation of works that previously needed more manual operations (jobs).

# Acknowledgments and Disclosure of Funding

We thank anonymous reviewers for their comments, as they have helped improving the paper very much. We declare no third party funding or support.

# References

[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, pages 1097–1105, 2012.   
[2] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–9, 2015.   
[3] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems 27, pages 3104–3112, 2014.   
[4] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30, pages 5998–6008, 2017.   
[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.   
[6] Yoshua Bengio, Andrea Lodi, and Antoine Prouvost. Machine learning for combinatorial optimization: a methodological tour d’ horizon. European Journal of Operational Research, 2020.   
[7] Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning. In ICLR (Workshop), 2017.   
[8] Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. In Advances in Neural Information Processing Systems 30, pages 6348–6358, 2017.   
[9] Chaitanya K Joshi, Thomas Laurent, and Xavier Bresson. An efficient graph convolutional network technique for the travelling salesman problem. arXiv preprint arXiv:1906.01227, 2019.   
[10] Wouter Kool, Herke van Hoof, and Max Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.   
[11] Yaoxin Wu, Wen Song, Zhiguang Cao, Jie Zhang, and Andrew Lim. Learning improvement heuristics for solving routing problems. arXiv preprint arXiv:1912.05784v2, 2019.   
[12] Paulo R de O da Costa, Jason Rhuggenaath, Yingqian Zhang, and Alp Akcay. Learning 2-opt heuristics for the traveling salesman problem via deep reinforcement learning. arXiv preprint arXiv:2004.01608, 2020.   
[13] Xinyun Chen and Yuandong Tian. Learning to perform local rewriting for combinatorial optimization. In Advances in Neural Information Processing Systems 32, pages 6281–6292, 2019.   
[14] André Hottung and Kevin Tierney. Neural large neighborhood search for the capacitated vehicle routing problem. arXiv preprint arXiv:1911.09539, 2019.   
[15] Hao Lu, Xingwen Zhang, and Shuang Yang. A learning-based iterative method for solving vehicle routing problems. In International Conference on Learning Representations, 2020.   
[16] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In Advances in Neural Information Processing Systems 28, pages 2692–2700. 2015.   
[17] Vijay R. Konda and John N. Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems 12, pages 1008–1014, 2000.   
[18] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In International Conference on Learning Representations, 2015.   
[19] MohammadReza Nazari, Afshin Oroojlooy, Lawrence Snyder, and Martin Takac. Reinforcement learning for solving the vehicle routing problem. In Advances in Neural Information Processing Systems 31, pages 9839–9849, 2018.   
[20] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, 1992.   
[21] Steven J Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. Selfcritical sequence training for image captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 7008–7024, 2017.   
[22] Bo Peng, Jiahai Wang, and Zizhen Zhang. A deep reinforcement learning algorithm using dynamic attention model for vehicle routing problems. arXiv preprint arXiv:2002.03282, 2020.   
[23] Hanjun Dai, Bo Dai, and Le Song. Discriminative embeddings of latent variable models for structured data. In International Conference on Machine Learning, pages 2702–2711, 2016.   
[24] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.   
[25] Michel Deudon, Pierre Cournut, Alexandre Lacoste, Yossiri Adulyasak, and Louis-Martin Rousseau. Learning heuristics for the tsp by policy gradient. In International conference on the integration of constraint programming, artificial intelligence, and operations research, pages 170–181. Springer, 2018.   
[26] Wikipedia. Anchoring (cognitive bias) — Wikipedia, the free encyclopedia. http://en. wikipedia.org/w/index.php?title $\ c =$ Anchoring%20(cognitive%20bias).   
[27] Ronald J. Williams and Jing Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3(3):241–268, 1991.   
[28] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free! In DeepRLStructPred@ICLR, 2019.   
[29] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In International Conference on Learning Representations, 2018.   
[30] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.   
[31] Diederik P. Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.   
[32] David L Applegate, Robert E Bixby, Vasek Chvatal, and William J Cook. The traveling salesman problem: a computational study. Princeton university press, 2006.   
[33] Keld Helsgaun. An effective implementation of the lin–kernighan traveling salesman heuristic. European Journal of Operational Research, 126(1):106–130, 2000.