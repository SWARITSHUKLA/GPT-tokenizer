import regex as re

class RegexTokenizer:

    def __init__(self, corpus,vocab_size):
        self.rergex_corpus = self.regex_list(corpus)
        self.vocab_size = vocab_size
        self.tokens = [strings.encode("utf-8") for strings in self.rergex_corpus] #raw bytes
        self.tokens = list(map(int, strings) for strings in self.tokens)
        self.merges = self.train()
        self.vocab_dict = self.vocab()

    def regex_list(self,corpus):
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        return (re.findall( GPT4_SPLIT_PATTERN, corpus))

    def get_stats(self, tokens):
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) +  1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i<len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i+=2
            else:
                newids.append(ids[i])
                i+=1
        return newids


    def train(self):
        num_merges = self.vocab_size-256
        ids_list = [list(smol_tokens) for smol_tokens in self.tokens]
        merges ={}
        for i in range (num_merges):
            combined_stats = {}
            for ids in ids_list:
                stats = self.get_stats(ids)
                for pair, count in stats.items():
                    combined_stats[pair] = combined_stats.get(pair, 0) +1
                
            if not combined_stats:
                break

            pair = max(combined_stats, key = combined_stats.get)
            idx = 256+i
            ids_list = [self.merge(ids,pair, idx) for ids in ids_list]
            merges[pair] = idx
        return merges
    
    def vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def encode(self, string):
        chunks = self.regex_list(string)
        all_tokens =[]
        for chunk in chunks:
            tokens = list(chunk.encode('utf-8'))
            while len(tokens)>=2:
                stats = self.get_stats(tokens)
                pair = min(stats, key = lambda p: self.merges.get(p, float('inf')))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                tokens = self.merge(tokens, pair, idx)
            all_tokens.extend(tokens)
        return all_tokens
        

    def decode(self, ids):
        tokens = b''.join(self.vocab_dict[idx] for idx in ids)
        text = tokens.decode('utf-8',  errors = 'replace')
        return text


m = RegexTokenizer('''Do you remember the times when we used to make LLMs count the occurrence of a specific letter in a word, like “How many r’s in strawberry?”

Back then, LLMs used to get it wrong a lot of times, but nowadays they don’t. Well, one of the factors behind it is the emergence of reasoning capabilities(The main reason behind it was The tokenization issue). It allows the model to reason through the problem. When this happened, the models could think like, “I have to count the number of r’s, so first let me break down the word into individual letters: S T R A W B E R R Y. Now let’s count the number of r’s sequentially: S is not an r, T is not an r, R is r, so the count becomes 1 …… “, This emergent property helped LLMs solve complex problems by breaking them down and thinking step by step.

In today’s time, we use GRPO. It stands for Group Relative Policy Optimization

Here is the abstract overview of what it does

Step 1 - It generates a few model responses.

Step 2 - It scores every model response.

Step 3 - It compares every response by the model relatively in the group and assigns a score.

Step 4 - It updates the model parameters using the advantages, so that the good responses become more likely

After having a quick overview, let’s begin with the Deep explanation

Sampling
Given an input, the model generates 
 number of outputs. The output of the model is represented by (
).

Advantage calculation
After getting the 
 number of outputs, we simply grade each of the outputs by a reward function or another model, and use an advantage function to calculate the Advantage value for each of the outputs. We simply take every output, subtract the mean, and divide it by the standard deviation (we calculate the mean and standard deviation using each of the outputs in a group)

After standardization, we can tell which responses are better than an average response: if 
 > 0, then the response is actually better than the average response; if 
 < 0, then the response is worse than the average response.

So we know which responses are better; now we have to update the model so that it produces better responses

Policy/Model Update
Before we begin this section, I would like to tell you that a policy is something that generates responses or takes action, so in our case, the policy is the Language model.

*If you can’t see the full equation you can scroll and see it.

The equation above can be broken down into its sub-pieces to make it more interpretable.

1 - The Probability Ratio
The equation above represents the probability ratio. Here, 
 Represents the language model’s policy, and 
 represents the parameters of the model.

 Represents the probability that the current model assigns to generating 
 given prompt 
.

The probability ratio is the probability assigned by the new model divided by the probability assigned by the old model new model (The difference between the old and te new models will be cleared in the example at the end).

If the probability ratio 
, then the model assigns a higher probability to the response 
 by the new model. If 
, then the model assigns a lower probability to the response 
 by the new model.

2 - The Clip function
The clip function prevents the model from changing itself too much; it doesn’t allow the probability ratio to go beyond the 
 range. The deviation is capped at 
 & 
 range.

3 - KL Divergence
The above equation is the KL divergence equation, i tells us how far off the probability distribution of the new model 
 is from the reference model 
.

The reference model is the model that we took right after SFT (Supervised Fine-Tuning) and before RL fine-tuning.

So the equation calculates how different the new model’s probabilities are from the reference model for those responses (
), and takes the expected value of that difference.

Walkthrough
Let’s say we take the batch size of 5, and each of the prompts in a batch contains 8 outputs of those responses, so the group size is 8, and the number of epochs is 3.

Step 1
We generate 5*8 = 40 responses.

Step 2
Then we calculate the Advantage of each of the prompt outputs, so 40 advantages each, we calculate Advantage for every example in the group independently.

Step 3
The minimum is between the unclipped surrogate objective 
 and the clipped surrogate objective 

Then we calculate the KL divergence function. Here, 
 is a hyperparameter that defines how strongly you want to penalize the model for deviating from the reference model

Finally, after having everything we need, we calculate the objective function and do a single backward pass. And we do it three times for a single batch because remember our number of epochs is 3.

NOTE - At the first epoch of every batch, 
, meaning both models are the same because we haven’t done any backward pass yet.''', 300)
print(m.decode(m.encode('lmao the 123world!! heheh')))