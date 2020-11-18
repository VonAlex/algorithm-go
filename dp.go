package leetcode

/*
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
	memo := make([]int, n+1)

	var _helper func(int) int
	_helper = func(n int) int {
		if n == 0 {
			return 0
		}
		if n == 1 || n == 2 {
			return 1
		}
		if memo[n] != 0 {
			return memo[n]
		}
		memo[n] = (_helper(n-1) + _helper(n-2)) % 1000000007
		return memo[n]
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

/*
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

/*
 * LeetCode T53. 最大子序和
 * https://leetcode-cn.com/problems/maximum-subarray/
 *
 * 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
 * 示例 1:
 * 输入: [-2,1,-3,4,-1,2,1,-5,4],
 * 输出: 6
 * 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
 */
// 方法 1：动态规划
func MaxSubArray(nums []int) int {
	numsLen := len(nums)
	// 1.定义状态：dp[i] 表示以 i 结尾子串的最大值
	dp := make([]int, numsLen)
	// 3.初始化：找到初始条件
	dp[0] = nums[0]
	for i := 1; i < numsLen; i++ {
		// 第 i 个子组合的最大值可以通过第 i-1 个子组合的最大值和第 i 个数字获得
		// 2.状态转移方程，如果 dp[i-1] 不能带来正增益的话，那么丢弃以前的最大值
		if dp[i-1] > 0 {
			dp[i] = dp[i-1] + nums[i]
		} else {
			// 抛弃前面的结果
			dp[i] = nums[i]
		}
	}
	max := nums[0]
	// 4.选出结果
	for _, sum := range dp {
		if sum > max {
			max = sum
		}
	}
	return max
}

// 动态规划的优化
func MaxSubArray2(nums []int) int {
	// 状态压缩：每次状态的更新只依赖于前一个状态，就是说 dp[i] 的更新只取决于 dp[i-1] ,
	// 我们只用一个存储空间保存上一次的状态即可。
	// var start, end, subStart, subEnd int  // 可以获得最大和子序列的边界位置
	sum := nums[0]
	maxSum := nums[0]
	numsLen := len(nums)
	for i := 1; i < numsLen; i++ {
		if sum > 0 {
			sum += nums[i]
			// subEnd++
		} else {
			sum = nums[i]
			// subStart = i
			// subEnd = i
		}
		if maxSum < sum {
			maxSum = sum
			// start = subStart
			// end = subEnd
		}
	}
	return maxSum
}

// 方法 2：Kadane算法
func MaxSubArray3(nums []int) int {
	sum := nums[0]
	maxSum := nums[0]
	for _, num := range nums {
		sum = maxInt(num, sum+num) // sum 能否提供正增益，与 dp 解法其实是一致的
		if maxSum < sum {
			maxSum = sum
		}
	}
	return maxSum
}

// 方法 3：分治法
// 时间复杂度 O(nlogn)
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

/*
 * LeetCode T70. 爬楼梯
 * https://leetcode-cn.com/problems/climbing-stairs/
 *
 * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
 * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
 * 注意：给定 n 是一个正整数。
 */
// 状态 n，n 阶台阶共有 f(n) 种不同的方法
// dp 状态转移方程，f(n) = f(n-1) + f(n-2)，一次只能走 1 或 2 个台阶，所以 f(n) 可以从 f(n-1) 到达，也可以从 f(n-2)到达
// 边界值 f(0) = 1, f(1) = 1
// 这个问题转换完跟斐波拉契数列是一样的，其他解法可以参考
func climbStairs(n int) int {
	if n == 0 || n == 1 {
		return 1
	}
	first, second := 1, 1
	res := 0
	for i := 2; i <= n; i++ {
		res = first + second
		first = second
		second = res
	}
	return res
}

// 方法 1：dp table，自底而上
// 状态:i，f(i) = max(f(i-1), nums[i]+f(i-2))
// 选择: 第 i 间房屋，偷还是不偷
// 偷窃第 i 间房屋，那么就不能偷窃第 i−1 间房屋，偷窃总金额为前 i−2 间房屋的最高总金额与第 i 间房屋的金额之和。
// 不偷窃第 i 间房屋，偷窃总金额为前 i−1 间房屋的最高总金额
// 边界条件：只有1间房屋，则偷窃该房屋; 只有两间房屋，选择其中金额较高的房屋进行偷窃
// 状态 -> 选择 -> 状态转移方程 -> 边界条件
func rob(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return nums[0]
	}
	dp := make([]int, n)
	// 边界条件
	dp[0] = nums[0]
	dp[1] = maxInt(nums[0], nums[1])
	for i := 2; i < n; i++ {
		dp[i] = maxInt(dp[i-1], nums[i]+dp[i-2])
	}
	return dp[n-1]
}

// 因为 dp[i] 只与dp[i-1] 和 dp[i-2] 有关，所以可以使用滚动数组存储前两次的结果，
// 使空间复杂度从 O(n) 降为 O(1)
func rob2(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return nums[0]
	}
	first := nums[0]
	second := maxInt(nums[0], nums[1])
	for i := 2; i < n; i++ {
		temp := maxInt(second, nums[i]+first)
		first = second
		second = temp
	}
	return second
}

// 方法 3：memo 备忘录，自顶而下
func rob3(nums []int) int {
	n := len(nums)
	memo := make([]int, n)
	for i := range memo {
		memo[i] = -1
	}
	var _dp func(int) int
	_dp = func(i int) int {
		if i >= n || i < 0 {
			return 0
		}
		if i == 0 {
			return nums[0]
		}
		if memo[i] != -1 {
			return memo[i]
		}
		res := maxInt(_dp(i-1), nums[i]+_dp(i-2))
		memo[i] = res
		return res
	}
	return _dp(n - 1)
}

/*
 * LeetCode T5. 最长回文子串
 * https://leetcode-cn.com/problems/longest-palindromic-substring/
 *
 * 给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
 */
// 方法 1：暴力法
// 枚举所有的子串 O(N^2) * 判断每个子串是否是回文串 O(N)
// 所以，时间复杂度为 O(N^3) ，空间复杂度为 O(1)

// 方法 2：动态规划
// 考虑奇偶字符串，枚举字符串的左右边界
// 时/空复杂度 O(N^2)
func longestPalindrome(s string) string {
	slen := len(s)
	if slen < 2 {
		return s
	}
	// 定义状态 dp[i][j] 表示子串 s[i..j] 是否为回文子串，闭区间，两端都能取到
	dp := make([][]bool, slen)
	for i := 0; i < slen; i++ {
		dp[i] = make([]bool, slen)
		dp[i][i] = true // 状态的初始化
	}
	start := 0
	maxLen := 1
	// 状态的转移 dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]
	// 看到 i + 1 和 j - 1 的坐标，就要考虑边界条件
	// j - 1 - (i + 1) + 1 = j - i -1
	// 如果 j - i - 1 == 0，即 s[i+1..j-1] 为空字符串，是回文
	// 如果 j - i - 1 == 1， 即 s[i+1..j-1] 为单字符串，也是回文
	// 所以要求 j - i - 1 < 2，即 j - i < 3

	// 填报，在本题的填表的过程中，只参考了左下方的数值。
	// 事实上可以优化，但是增加了代码编写和理解的难度，丢失可读和可解释性。在这里不优化空间
	for j := 1; j < slen; j++ {
		for i := 0; i < j; i++ { // s[i..j]
			if s[i] != s[j] {
				dp[i][j] = false
				continue
			}
			if j-i < 3 {
				dp[i][j] = true
			} else { // s[i] == s[j], s[i..j] 是否是回文串由 s[i+1..j-1] 决定
				dp[i][j] = dp[i+1][j-1]
			}
			if dp[i][j] && j-i+1 > maxLen { // 记录下最长子串的 start
				start = i
				maxLen = j - i + 1
			}
		}
	}
	return s[start : start+maxLen]
}

// 方法 3：中心扩散法
// 时间复杂度为 O(N^2)，空间复杂度为 O(1)
// 考虑奇偶字符串，枚举回文子串的“中心位置”，这样的位置共有 2n-1 个
//   ↓ ↓
// a b b a 偶数字符串
//   ↓
// a b a  奇数字符串
func longestPalindrome2(s string) string {
	slen := len(s)
	if slen < 2 {
		return s
	}
	var start, end int
	// 最后一个无法扩散，因此索引最大等于 len-2
	for i := 0; i < slen-1; i++ {
		l1, r1 := expendArountCenter(s, i, i)
		if r1-l1 > end-start {
			start, end = l1, r1
		}
		l2, r2 := expendArountCenter(s, i, i+1)
		if r2-l2 > end-start {
			start, end = l2, r2
		}
	}
	return s[start : end+1]
}

// 以 l/r 为中心，往两边扩散，直到遇到非回文串
func expendArountCenter(s string, l, r int) (int, int) {
	slen := len(s)
	i, j := l, r
	for i >= 0 && j < slen {
		if s[i] != s[j] {
			break
		}
		i--
		j++
	}
	return i + 1, j - 1
}
