package leetcode

/**
 * LeetCode T509. 斐波那契数
 * https://leetcode-cn.com/problems/fibonacci-number/
 * 面试题10- I. 斐波那契数列
 * https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/
 * 斐波那契数，通常用 F(n) 表示，形成的序列称为斐波那契数列。
 * 该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。
 * 给定 N，计算 F(N)。
 */
// 方法 1：递归法
// 时间复杂度 O(N^2)
// 由于存在大量重复计算，会报错“超出时间限制”
func Fib(n int) int {
	if n == 0 {
		return 0
	}
	if n == 1 || n == 2 {
		return 1
	}
	return (Fib(n-1) + Fib(n-2)) % 1000000007
}

// 方法 2：备忘录递归
// 上面方法中存在的重复计算问题，可以使用使用一个数组存放已经计算过的值,
// 当数组中有的话，直接从数组中取到，否则进行递归运算
// 时间复杂度和空间复杂度为 O(N)
func Fib2(n int) int {
	res := make([]int, n+1)

	var _helper func(int) int
	_helper = func(n int) int {
		if n == 0 {
			return 0
		}
		if n == 1 || n == 2 {
			return 1
		}
		if res[n] != 0 {
			return res[n]
		}
		res[n] = (_helper(n-1) + _helper(n-2)) % 1000000007
		return res[n]
	}
	return _helper(n)
}

// 方法 3：动态规划
// 状态定义： 设 dp 为一维数组，其中 dp[i] 的值代表斐波那契数列第 i 个数字
// 转移方程： dp[i] = dp[i-1] + dp[i-2] ，即对应数列定义 f(n) = f(n-1) + f(n-2)
// 初始状态： dp[0] = 0 dp[1] = 1 ，即初始化前两个数字
// 返回值： dp[n] ，即斐波那契数列的第 n 个数字
// 时间复杂度 O(N)，空间复杂度为 O(1)
func Fib3(n int) int {
	if n == 0 {
		return 0
	}
	dp := make([]int, n+1)
	dp[0], dp[1] = 0, 1
	for i := 2; i <= n; i++ {
		dp[i] = (dp[i-1] + dp[i-2]) % 1000000007
	}
	return dp[n]
}

// 方法 4
// 进一步优化
// f(n) 只跟 f(n-1) 和 f(n-2) 有关，所以只需要保留前两个值，以及和值，不断进行迭代
// 设正整数 x, y, p，求余符号为 ⊙，那么(x + y) ⊙ p = (x ⊙ p + y ⊙ p) ⊙ p
// 时间复杂度 O(N)，空间复杂度为 O(1)
func Fib4(n int) int {
	a, b := 0, 1
	sum := 0
	for i := 0; i < n; i++ { // 需要 n-1 次循环
		sum = (a + b) % 1000000007
		a = b
		b = sum
	}
	return a
}

// func maxSubArray(nums []int) int {

// }
