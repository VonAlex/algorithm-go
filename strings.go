package leetcode

import (
	"math"
	"strconv"
)

/**
 * LeetCode 题 3 无重复字符的最长子串
 * https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
 *
 * 给定一个字符串，请你找出其中不含有重复字符的最长子串的长度
 *
 * 关键点在左端字符位置的更新
 */
// 1. 暴力法，但由于时间限制，会出现 TLE
// 逐个检查所有的子字符串，看它是否不含有重复的字符。

// 2. 优化版的滑动窗口
// 时间复杂度 O(n)，空间复杂度 O(min(m,n))，m 是字符集的大小
func LengthOfLongestSubstring(s string) int {
	var ans, left, length int
	indexs := make(map[rune]int) // 定义字符到索引的映射
	for i, c := range s {
		pos, ok := indexs[c]

		// 如果 s[j] 在 [i, j) 范围内有与 j' 重复的字符。
		// 不需要逐渐增加 i，可直接跳过 [i，j'] 范围内的所有元素，并将 i 变为 j' + 1。
		if ok && left < pos { // 遇到重复字符时才可能更新 left
			left = pos // left = max(left, pos)
		}
		length = i - left + 1
		if ans < length { // ans = max(length, ans)
			ans = length
		}
		indexs[c] = i + 1 // c 的 value 保存为当前位置的下一个，方便更新 left
	}
	return ans
}

/**
 * 剑指 offer 面试题 05 替换空格
 * 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
 * https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/
 *
 * 输入：s = "We are happy."
 * 输出："We%20are%20happy."
 */

// 思路：从后往前挪动位置，时间复杂度和空间复杂度均为 O(n)
func ReplaceSpace(s string) string {

	// 首先计算出空格的个数，方便后面计算结果字符串的长度
	spacenums := 0
	for _, char := range s {
		if char == ' ' {
			spacenums++
		}
	}
	slen := len(s)
	res := make([]byte, slen+spacenums*2)
	end := len(res) - 1
	for i := slen - 1; i >= 0; i-- {
		if s[i] == ' ' {
			res[end] = '0'
			end--
			res[end] = '2'
			end--
			res[end] = '%'
		} else {
			res[end] = s[i]
		}
		end--
	}
	return string(res)
}

/**
 * 剑指 offer 面试题 50. 第一个只出现一次的字符
 * https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/
 * s = "abaccdeff"
 * 返回 "b"
 * s = ""
 * 返回 " "
 * 限制：0 <= s 的长度 <= 50000
 */

// 解法 1 暴力遍历，时间复杂度 O(n^2)

// 解法 2-1 两次遍历法（map）
// 时间复杂度为 O(n)
func FirstUniqChar(s string) byte {
	if s == "" {
		return ' '
	}
	// 使用 map 计数
	checkSet := make(map[rune]int)
	for _, si := range s {
		checkSet[si]++
	}
	for _, si := range s {
		if checkSet[si] == 1 {
			return byte(si)
		}
	}
	return ' '
}

// 解法 2-2 两次遍历法（数组）
// 如果 s 中不仅仅是小写字母，那么只能用 map
func FirstUniqChar2(s string) byte {
	if s == "" {
		return ' '
	}
	// 使用数组计数
	checkSet := make([]int, 26)
	for _, si := range s {
		checkSet[si-'a']++
	}

	for _, si := range s {
		if checkSet[si-'a'] == 1 {
			return byte(si)
		}
	}
	return ' '
}

/**
 * LeetCode 题 7 整数反转
 * https://leetcode-cn.com/problems/reverse-integer/
 *
 * 输入: 123  输出: 321
 * 输入: -123 输出: -321、
 * 输入: 120  输出: 21
 *
 * 假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−231,  231 − 1]。
 * 请根据这个假设，如果反转后整数溢出那么就返回 0。
 */

// 解法 1：res = res*10 + pop
func Reverse(x int) int {
	res := 0
	for x != 0 {
		pop := x % 10
		// res = res*10 + pop
		// if res > math.MaxInt32 || res < math.MinInt32 {
		// 	return 0
		// }
		// x /= 10

		x /= 10

		// 7 是因为 2^31 - 1 = 2147483647，个位数是 7
		if res > math.MaxInt32/10 || (res == math.MaxInt32/10 && pop > 7) {
			return 0
		}
		// 8 是因为 (-2)^31 = -2147483648，个位数是 8
		if res < math.MinInt32/10 || (res == math.MinInt32/10 && pop < -8) {
			return 0
		}
		res = res*10 + pop
	}
	return res
}

// 解法 2：先转成字符串，然后反转字符串，再转成数字
func Reverse2(x int) int {
	sign := 1
	if x < 0 {
		sign = 0 - sign
		x = 0 - x
	}
	s := strconv.Itoa(x)
	bts := make([]byte, 0, len(s))
	for i := len(s) - 1; i >= 0; i-- {
		bts = append(bts, s[i])
	}
	// Atoi 会自动处理 '020' 这种首字母为 0 的数字
	res, err := strconv.Atoi(string(bts))
	if err != nil {
		return 0
	}
	res *= sign
	if res > math.MaxInt32 || res < math.MinInt32 {
		return 0
	}
	return res
}
