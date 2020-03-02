package leetcode

/**
 * LeetCode 题 3 无重复字符的最长子串
 * https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
 *
 * 给定一个字符串，请你找出其中不含有重复字符的 最长子串的长度
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
