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
// 方法 1：递归法（暴力解）
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

// 方法 2：备忘录递归（自顶向下）
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

// 方法 3：动态规划（自底向上）
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
	// 初始状态
	dp[0], dp[1] = 0, 1
	for i := 2; i <= n; i++ {
		// 转移方程
		dp[i] = (dp[i-1] + dp[i-2]) % 1000000007
	}
	// 返回值
	return dp[n]
}

// 方法 4
// 进一步优化
// f(n) 只跟 f(n-1) 和 f(n-2) 有关，所以只需要保留前两个值，以及和值，不断进行迭代
// 设正整数 x, y, p，求余符号为 ⊙，那么(x + y) ⊙ p = (x ⊙ p + y ⊙ p) ⊙ p
// 时间复杂度 O(N)，空间复杂度为 O(1)
func Fib4(n int) int {
	if n == 0 {
		return 0
	}
	if n == 1 || n == 2 {
		return 1
	}
	sum := 0
	prev, curr := 1, 1
	for i := 3; i <= n; i++ {
		sum = (prev + curr) % 1000000007
		prev = curr
		curr = sum
	}
	return curr
}

/**
 * LeetCode T322. 零钱兑换
 * https://leetcode-cn.com/problems/coin-change/
 *
 * 给定不同面额的硬币 coins 和一个总金额 amount。
 * 编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1
 * 示例 1:
 * 输入: coins = [1, 2, 5], amount = 11
 * 输出: 3
 * 解释: 11 = 5 + 5 + 1
 *
 * 示例 2:
 * 输入: coins = [2], amount = 3
 * 输出: -1
 *
 * 说明:你可以认为每种硬币的数量是无限的。
 */
// 方法 1：备忘录递归（自顶向下）
// count 数组保存已经计算过的 amount
// 假设 N 为金额，M 为面额数
// 时间复杂度 = 子问题数目 * 每个子问题的时间，所以为 O(MN)
// 因为有一个备忘录数组，所以空间复杂度为 O(N)
func CoinChange(coins []int, amount int) int {
	// 状态定义
	count := make([]int, amount+1)
	var _dp func(int) int
	_dp = func(n int) int {
		if n == 0 { // amount 为 0，需要 0 个硬币
			return 0
		}
		// 查看备忘录，避免重复计算
		if count[n] != 0 {
			return count[n]
		}
		// 凑成 amount 金额的硬币数最多只可能等于 amount(全部用 1 元面值的硬币)
		// 初始化为 amount+1就相当于初始化为正无穷，便于后续取最小值
		count[n] = amount + 1
		for _, coin := range coins {
			if n-coin < 0 { // 子问题无解跳过
				continue
			}
			// 记入备忘录，转移方程
			count[n] = min(count[n], _dp(n-coin)+1)
		}
		return count[n]
	}
	res := _dp(amount)
	if res == amount+1 {
		return -1
	}
	return res
}

// 方法 2：动态规划（自下而上）
// 时间复杂度 O(MN)，空间复杂度为 O(N)
// 将方法 1 中的递归转换成迭代
func CoinChange2(coins []int, amount int) int {
	dpLen := amount + 1
	dp := make([]int, dpLen)
	for n := 1; n < dpLen; n++ {
		dp[n] = amount + 1
		for _, coin := range coins {
			if n-coin < 0 {
				continue
			}
			dp[n] = min(dp[n], dp[n-coin]+1)
		}
	}
	if dp[amount] == amount+1 {
		return -1
	}
	// 省去了 amount = 0 的检查，初始值 dp[0] = 0
	return dp[amount]
}

// func maxSubArray(nums []int) int {

// }

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
