# Cost Optimization Lab

## Introduction

Let's explore the following scenario:

You are currently a startup customer who is building an AI solution for detecting pill shapes during the manufacturing process. Your developers have been hard at work on the AWS Cloud setting up the infrastructure that's necessary to train the AI model that you plan to use. They are using services like Sagemaker, ParallelCluster and Batch. However, there is a growing concern internally within your organization that the cost of training this model could become more expensive than initially thought. Rather than try to raise more funds, you've been tasked with coming up with ways to reduce the cost of training your model on the AWS Cloud.

There are a number of different ways you could potentially reduce this cost. However, making that decision is often a tradeoff that depends on what your specific scenario calls for. Let's explore some of those scenarios below:

## Scenario 1: Your Model Training Job is Fault Tolerant

After an initial conversation with your development team, you realized that your training job has been designed to be fault tolerant. As a result, if any of your individual compute nodes is interrupted during the training process, that node checkpoints its current state before being terminated. Because of this check pointing, other nodes are easily able to pick up where the interrupted node left off.

Separately, you had a conversation with your executive team and they don't care how long the training job takes as long as its completed for the lowest possible cost.

In this scenario, how should you proceed?

### Spot Instances

Spot Instances are spare EC2 capacity that can save you up to 90% off of On-Demand prices that AWS can interrupt with a 2-minute notification.

Unlike On-demand EC2 instance pricing, spot instance pricing is dynamic and based on long term supply and demand trends for each Spot Capacity Pool. Spot capacity pools are combinations of EC2 instance families, sizes and availability zones. For example, the current spot price you pay for a G6.xlarge running in az-1 in US-East-1 is different than the spot price you pay for for a G6.xlarge running in az-2 in US-East-2. This raises the question of how you can implement logic to ensure that you are running spot instances from the lowest cost spot capacity pools.

However, price isn't the only part of the equation. Another important thing to understand when using spot instances is the fact that they can be interrupted. Therefore, when using spot instances the question often becomes "How can I run spot instances for as low of a spot price as possible WHILE minimizing the likelihood of spot interruption?" The answer is by being as diversified as possible AND using the appropriate `allocation strategy`.

### Diversification

When using spot instances, being able to provision instances from a greater number of spot capacity pools will increase the likelihood that your workload can be completed with minimal interruption. For example, if you test your workload on a G6.xlarge and want to exclusively use G6.xlarge instances across the six different availability zones in US-East-1, you are able to provision instances from six different spot capacity pools. You can calculate this number by multiplying the number of instances and availability zones your workload is able to provision. 6 AZs * 1 instance type = 6 Spot capacity pools. Six spot capacity pools is a good start but in an ideal situation, you want to be able to use as many spot capacity pools as possible.

In order to increase the number of spot capacity pools we can use, we can use the concept of diversification. Diversification means being flexible and using as many instance types and availability zones as possible that can support our workload. There are a number of different types of flexibility we can use, including the following:

- **Size**: If we update our configuration to provision G6.2xlarges and G6.4xlarges in addition to G6.xlarges, we are now able to take advantage of 18 different spot capacity pools. (3 instance types * 6 AZs = 18 Spot capacity pools).
- **Instance Family**: If we update our configuration to provision G5.xlarges, G5.2xlarges and G5.4xlarges, we are now able to take advantage of 36 Spot capacity pools (6 instance types * 6 AZs = 36 Spot capacity pools).
- **Region**: If we are able to run our workload within either US-East-1 (6 AZs) OR US-East-2 (3 AZs) with our current config, we are now able to take advantage of a total of 54 Spot capacity pools (6 instance types * 9 AZs = 54 Spot capacity pools).
- **Time**: Spot capacity is inherently excess EC2 capacity. By running your workload at night or over the weekend you can reduce the likelihood of interruption.

### Allocation Strategy

Now that we have 54 Spot Capacity pools, how do we figure out which of those capacity pools we want to provision an instance from? The answer is `allocation strategy`.

The `allocation strategies` that you can use depend on the service that your workload is using. In the case of ParallelCluster, you can take advantage of the following 3 allocation strategies: `lowest-price` | `capacity-optimized` | `price-capacity-optimized`. Lowest Price is the default.

- **`lowest-price`**: Spot instances will be provisioned from the spot capacity pool with the lowest current spot price. We don't typically recommend using this strategy because while it will provide the lowest possible price, this strategy doesn't take into account the likelihood that the instance will be interrupted.
- **`capacity-optimized`**: Spot instances will be provisioned from the spot capacity pool with the most excess spot capacity.
- **`price-capacity-optimized`**: Spot instances will be provisioned from the spot capacity pool with a combination of the lowest price and most excess spot capacity. This is the strategy we typically recommend since it will take into account both price and availability, whereas the other two strategies will only take either price OR availability into account, not both.

Further information on available spot allocation strategies in AWS Batch and AWS ParallelCluster can be found at the following links:
- [AWS Batch Allocation Strategies](https://docs.aws.amazon.com/batch/latest/userguide/allocation-strategies.html)
- [AWS ParallelCluster Allocation Strategies](https://docs.aws.amazon.com/ParallelCluster/latest/ug/Scheduling-v3.html#yaml-Scheduling-SlurmQueues-AllocationStrategy)

## Conclusion

The above served as an introduction to best practices for leveraging GPU based spot instances. For further information on more spot best practices, see the [official AWS documentation on spot best practices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html).
