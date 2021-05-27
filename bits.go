package leetcode

import (
	"math/bits"
)

// 计算一个数二进制表示中有多少个 1
func HammingWeight(n int) int {
	count := 0
	for n != 0 {
		count++
		n = n & (n - 1)
	}
	return count
}

/*
 * LC 461. 汉明距离
 * https://leetcode-cn.com/problems/hamming-distance/
 *
 * 两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
 * 给出两个整数 x 和 y，计算它们之间的汉明距离。
 */
func hammingDistance(x int, y int) int {
	var ans int
	s := x ^ y
	for s > 0 {
		ans += s & 1
		s >>= 1
	}
	return ans
}

func hammingDistance2(x int, y int) int {
	return bits.OnesCount(uint(x ^ y))
}

/*
 * LeetCode T231. 2的幂
 * https://leetcode-cn.com/problems/power-of-two/
 *
 * 给定一个整数，编写一个函数来判断它是否是 2 的幂次方。
 */
// 提示：如果一个数是 2 的幂次方，那么它的二进制表示一定只含有 1 个 1

// 方法 1：迭代不断地除 2，符合条件的结果是 1
func isPowerOfTwo(n int) bool {
	if n <= 0 {
		return false
	}
	for n%2 == 0 {
		n /= 2
	}
	return n == 1
}

// 方法 2：位运算：去除二进制中最右边的 1
// 使用 n&(n-1) 可以把 n 最右边的 1 抹掉，所以符合条件时，结果为 0
func isPowerOfTwo2(n int) bool {
	if n <= 0 {
		return false
	}
	return n&(n-1) == 0
}

// 方法 3：位运算：获取二进制中最右边的 1
// 在补码表示法中，-n = ~n + 1, n 的按位取反加 1
// 2 的幂次方二进制表示只有一个 1，取到这个 1，跟原数值相等
func isPowerOfTwo3(n int) bool {
	if n <= 0 {
		return false
	}
	return n&(-n) == n
}

// 判断两个数是否异号
// 使用 x*y 有可能造成数值溢出
func isOppositeSign(x, y int) bool {
	return (x ^ y) < 0
}

/*
 * LeetCode 面试题56 - I. 数组中数字出现的次数
 * https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/
 *
 * 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。
 * 要求时间复杂度是O(n)，空间复杂度是O(1)。
 * 示例 1：
 * 	   输入：nums = [4,1,4,6]
 *	   输出：[1,6] 或 [6,1]
 */
// 分组异或
// 思路：如果除了一个数字以外，其他数字都出现了两次，那么如何找到出现一次的数字？
// 就是全员做亦或操作，相同的数字两两抵消，0 ^ a = a
// 本题有 2 个数字，那么可以先办法把这组数分成两堆，每一堆里包含一个唯一的数，这样问题简化了以前那个题目了
// 那么如果去分成 2 堆呢？
// 假设两个只出现一次的数是 a 和 b，那么全员异或结果实际上就是 a ^ b
// 结果中取任一个二进制为 1 的位，表示 a 和 b 在这个位置上不同。
// 所以，可以根据这个 1 的位置来把数据划成 2 堆
func SingleNumbers(nums []int) []int {
	if len(nums) == 0 {
		return nums
	}
	ret := 0
	for _, num := range nums {
		ret ^= num
	}
	div := 1
	for ret&div == 0 {
		div <<= 1 // 找二进制表示中为 1 的最右位置
	}
	var a, b int
	for _, num := range nums {
		if num&div == 0 {
			a ^= num
		} else {
			b ^= num
		}
	}
	return []int{a, b}
}

/*
 * LC 1486. 数组异或操作
 * https://leetcode-cn.com/problems/xor-operation-in-an-array/
 */
func xorOperation(n int, start int) int {
	var res int
	for i := 0; i < n; i++ {
		n := start + 2*i
		res ^= n
	}
	return res
}

/*
 * LC 1734. 解码异或后的排列
 * https://leetcode-cn.com/problems/decode-xored-permutation/
 */
func decode(encoded []int) []int {

	// encoded 长度为 n - 1， perm 长度为 n
	n := len(encoded) + 1
	var total int
	for i := 1; i <= n; i++ {
		total ^= i
	}

	var odd int
	for i := 1; i < n-1; i += 2 {
		odd ^= encoded[i]
	}

	perm := make([]int, n)
	perm[0] = total ^ odd

	for i := 0; i < n-1; i++ {
		perm[i+1] = encoded[i] ^ perm[i]
	}
	return perm
}

/*
 * LC 1310. 子数组异或查询
 * https://leetcode-cn.com/problems/xor-queries-of-a-subarray/
 */
// 解法 1：时间复杂度 O(mn)
func xorQueries(arr []int, queries [][]int) []int {
	rows := len(queries)
	res := make([]int, 0, rows)
	if rows == 0 {
		return res
	}
	arrLen := len(arr)
	for i := 0; i < rows; i++ {
		var elem int
		l, r := queries[i][0], queries[i][1]
		for j := l; j <= r; j++ {
			if j >= arrLen {
				return res
			}
			elem ^= arr[j]
		}
		res = append(res, elem)
	}
	return res
}

// 解法 2：前缀和，时间复杂度 O(n)
func xorQueries2(arr []int, queries [][]int) []int {
	xors := make([]int, len(arr)+1)
	for i, v := range arr {
		xors[i+1] = xors[i] ^ v
	}
	res := make([]int, len(queries))
	for i, p := range queries {
		res[i] = xors[p[0]] ^ xors[p[1]+1]
	}
	return res
}

func countTriplets(arr []int) int {
	n := len(arr)
	xors := make([]int, n+1)
	for i, v := range arr {
		xors[i+1] = xors[i] ^ v
	}

	var ans int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			for k := j; k < n; k++ {
				if xors[i] == xors[k+1] {
					ans++
				}
			}
		}
	}
	return ans
}

func countTriplets2(arr []int) int {
	n := len(arr)
	xors := make([]int, n+1)
	for i, v := range arr {
		xors[i+1] = xors[i] ^ v
	}

	var ans int
	for i := 0; i < n; i++ {
		for k := i + 1; k < n; k++ {
			if xors[i] == xors[k+1] {
				ans += (k - i)
			}
		}
	}
	return ans
}

func countTriplets3(arr []int) int {
	n := len(arr)
	xors := make([]int, n+1)
	for i, v := range arr {
		xors[i+1] = xors[i] ^ v
	}

	var ans int
	cnt := make(map[int]int)
	total := make(map[int]int)
	for k := 0; k < n; k++ {
		if m, has := cnt[xors[k+1]]; has {
			ans += (m*k - total[xors[k+1]])
		}
		cnt[xors[k]]++
		total[xors[k]] += k
	}
	return ans
}

func countTriplets4(arr []int) int {
	var ans, s int
	cnt := make(map[int]int)
	total := make(map[int]int)
	for k, v := range arr {
		prev := s
		s ^= v
		if m, has := cnt[s]; has {
			ans += (m*k - total[s])
		}
		cnt[prev]++
		total[prev] += k
	}
	return ans
}
