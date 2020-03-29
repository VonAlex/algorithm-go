package leetcode

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
