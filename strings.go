package leetcode

/**
 * 题3 两数之和
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
