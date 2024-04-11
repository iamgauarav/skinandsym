l = list(map(int, input().split()))

maxi = 0
k = len(l) - 1
left = 0

while left < k:
    maxi = max(maxi, l[left] + l[k])
    left += 1
    k -= 1

print(maxi)
