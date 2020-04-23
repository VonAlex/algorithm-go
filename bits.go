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

// 方法 1：位运算：去除二进制中最右边的 1
// 使用 n&(n-1) 可以把 n 最右边的 1 抹掉，所以符合条件时，结果为 0
func IsPowerOfTwo(n int) bool {
	if n <= 0 {
		return false
	}
	return n&(n-1) == 0
}

// 方法 2：位运算：获取二进制中最右边的 1
// 在补码表示法中，-n = ~n + 1, n 的按位取反加 1
// 2 的幂次方二进制表示只有一个 1，取到这个 1，跟原数值相等
func IsPowerOfTwo2(n int) bool {
	if n <= 0 {
		return false
	}
	return n&(-n) == n
}
