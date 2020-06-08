package leetcode

// 计算一个数二进制表示中有多少个 1
func HammingWeight(n int) int {
	count := 0
	for n != 0 {
		count++
		n = n & (n - 1)
	}
	return count
}

/**
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

/**
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
