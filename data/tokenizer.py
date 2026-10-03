from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []
        for _ in range(num_merges):
            freq = {}
            for i in range(len(tokens) - 1):
                pairs = (tokens[i], tokens[i + 1])
                freq[pairs] = freq.get(pairs, 0) + 1
            if not freq: break
            
            # tie breaker using lambda function
            best = min(freq, key=lambda p: (-freq[p], p))
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else: 
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            merges.append(list(best))
        return merges