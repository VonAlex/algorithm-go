package leetcode

import (
	"math"
)

/**
 * LeetCode 面试题16. 数值的整数次方
 * https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/
 *
 * 实现函数 double Power(double base, int exponent)，求 base 的 exponent 次方。不得使用库函数，同时不需要考虑大数问题。
 *
 * 示例 1:
 *        输入: 2.00000, 10
 *        输出: 1024.00000
 */
func MyPow(x float64, n int) float64 {
	// 0 的负整数次方，是没有意义的
	if floatEqual(x, 0.0) && n < 0 {
		return 0.0
	}
	sign := 1
	if n < 0 {
		sign = -1
		n = -n
	}
	res := powWithUnsignedEx(x, n)
	// 指数为负数，求倒数
	if sign < 0 {
		res = 1.0 / res
	}
	return res
}

// 浮点型数值比较大小 |x - y| < 0.0000001
func floatEqual(x, y float64) bool {
	if x-y > -0.0000001 && x-y < 0.0000001 {
		return true
	}
	return false
}

/*
 * 求一个数的正整数次方
 * 方法 1：二分法，将 n 对半折，遇到奇数次方要格外处理
 * 当 n 为偶数时，x^n = x^(n/2) * x^(n/2)
 * 当 n 为奇数时，x^n = x^(n/2) * x^(n/2) * x
 * 其中 n/2 又可以根据上面的公式同等求解
 * 时间复杂度为 O(logN)
 */
func powWithUnsignedEx(x float64, n int) float64 {
	if n == 0 { // 0 次方返回 1
		return 1
	}
	if n == 1 { // 1 次方返回本身
		return x
	}
	res := powWithUnsignedEx(x, n>>1)
	res *= res
	if n&1 == 1 { // 遇到奇数
		res *= x
	}
	return res
}

/*
 * 方法 2：位运算法
 * 以 x^10 为例, 10 的二进制表示是 1010
 * 所以 x^10 = x^8 * x^2
 */
func powWithUnsignedEx2(x float64, n int) float64 {
	res := 1.0
	accum := x
	for n != 0 {
		if n&1 == 1 { // 只有二进制最末的比特位是 1，才将 accum 类乘到 res里
			res *= accum
		}
		accum *= accum // 不断求平方值，x → x^2 → x^4 → x^8
		n = n >> 1
	}
	return res
}

/*
 * LeetCode 面试题17. 打印从1到最大的n位数
 * https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/
 *
 * 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
 *
 * 示例 1:
 *        输入: n = 1
 *        输出: [1,2,3,4,5,6,7,8,9]
 */
// 其实这道题被 LeetCode 做简单的，应有的考虑 n 过大溢出情况，这里根本没有考虑到
func PrintNumbers(n int) []int {
	max := int(math.Pow10(n)) - 1
	res := make([]int, max)
	for i := 1; i <= max; i++ {
		res[i-1] = i
	}
	return res
}

// 大数问题，字符串表示
// 使用全排列的解法，每个数位上的数都是数字 0-9
func PrintNumbers2(n int) []string {
	number := make([]byte, n)
	res := []string{}
	for i := 0; i < 10; i++ {
		number[0] = byte(i) + '0' // 最末位
		resub := printNumbersHelper(number, n, 0)
		res = append(res, resub...)
	}
	// TODO：把 res 每一项进行字符串反转得到最终结果
	return res
}

func printNumbersHelper(number []byte, length, index int) []string {
	res := []string{}
	if index == length-1 {
		res = append(res, string(number))
		return res
	}
	for i := 0; i < 10; i++ {
		number[index+1] = byte(i) + '0'
		resub := printNumbersHelper(number, length, index+1) // 下一位
		res = append(res, resub...)
	}
	return res
}

/*
 * LeetCode T7. 整数反转
 * https://leetcode-cn.com/problems/reverse-integer/
 *
 * 给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转
 *
 * 示例 1:
 *     输入: 123
 *     输出: 321
 * 假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−2^31,  2^31 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。
 */
// 时间复杂度 O(log(x)), 空间复杂度 O(1)
func reverseInt(x int) int {
	res := 0
	for x != 0 {
		remain := x % 10
		// [-2147483648, 2147483647]
		if res > math.MaxInt32/10 || (res == math.MaxInt32/10 && remain > 7) {
			return 0
		}
		if res < math.MinInt32/10 || (res == math.MinInt32/10 && remain < -8) {
			return 0
		}
		res = res*10 + remain
		x /= 10
	}
	return res
}

/*
 * LeetCode T9. 回文数
 * https://leetcode-cn.com/problems/palindrome-number/
 *
 * 判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
 *
 * 示例 1:
 *     输入: 121
 *     输出: true
 *
 * 示例 2:
 *     输入: -121
 *     输出: true
 *     解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
 */

// 方法1：数学解法
// 分别得到最高位与最低位进行比较，以 1221 为例
func isPalindromeInt(x int) bool {
	if x < 0 {
		return false
	}

	// div = 1000 用于求最高位
	div := 1
	for y := x; y >= 10; y /= 10 {
		div *= 10
	}
	for x > 0 {
		left := x / div // 1
		right := x % 10 // 1
		if left != right {
			return false
		}
		x %= div   // 去头 221
		x /= 10    // 去尾 22
		div /= 100 // 去掉了两位数字，除数 / 100
	}
	return true
}

// 方法2：反转一半数字
func isPalindromeInt2(x int) bool {
	if x < 0 || (x != 0 && x%10 == 0) {
		return false
	}
	rev := 0

	for x > rev {
		rev = rev*10 + x%10
		x /= 10
	}
	return x == rev || x == rev/10
}

// 方法3：反转全部数字
func isPalindromeInt3(x int) bool {
	if x < 0 {
		return false
	}
	rex := reverseInt(x)
	return x == rex
}

// 方法4：字符串法
