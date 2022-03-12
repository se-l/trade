def sherlockAndAnagrams(s):
    from collections import Counter
    from itertools import combinations
    # a = [Counter(list(s[j:j+i])) for i in range(1, len(s), 1) for j in range(len(s)+1-i)]
    a = [''.join(sorted([k + str(v) for k, v in Counter(list(s[j:j + i])).items()])) for i in range(1, len(s), 1) for j in range(len(s) + 1 - i)]
    a = Counter(a)
    return sum([len(list(combinations(range(v), 2))) for v in a.values()])
    anagrams = 0
    for i in range(len(a)-1):
        for j in range(i+1, len(a)):
            if a[i] == a[j]:
                anagrams += 1
    return anagrams
    a = [''.join(sorted([k+str(v) for k, v in Counter(list(s[j:j+i])).items()])) for i in range(1, len(s), 1) for j in range(len(s)+1-i)]
    for v in b.values():
        if v == 1:
            continue
        anagrams += v#math.factorial(v-1)
    return anagrams

# print(sherlockAndAnagrams('abba'))
print(sherlockAndAnagrams('cdcd'))
