package leetcode

import (
	"math"
	"strconv"
	"unicode"
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

/**
 * leetcode 题 125 验证回文串
 * https://leetcode-cn.com/problems/valid-palindrome/
 * 给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写
 * 说明：本题中，我们将空字符串定义为有效的回文串
 *
 * 输入: "A man, a plan, a canal: Panama"
 * 输出: true
 *
 *
 */
func IsPalindrome(s string) bool {
	l := 0
	r := len(s) - 1
	valid := func(v rune) bool {
		return unicode.IsDigit(v) || unicode.IsLetter(v)
	}
	for l < r {
		lletter := unicode.ToLower(rune(s[l]))
		rletter := unicode.ToLower(rune(s[r]))
		if !valid(lletter) {
			l++
			continue
		}
		if !valid(rletter) {
			r--
			continue
		}
		if lletter != rletter {
			return false
		}
		l++
		r--
	}
	return true
}

/*
 * leetcode 题 131 分割回文串
 * https://leetcode-cn.com/problems/palindrome-partitioning/
 * 给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
 * 返回 s 所有可能的分割方案。
 * 示例:
 * 输入: "aab"
 * 输出:
 * [
 *   ["aa","b"],
 *   ["a","a","b"]
 * ]
 *
 * 以 aabb 为例
 * 解法 1：分治法
 * 先考虑在第 1 个位置切割, a | abb
 * abb 的所有结果是 [a, bb], [a, b, b]
 * 然后考虑在第 2 个位置切割, aa | bb
 * bb 的所有结果是 [bb], [b, b]
 * 以此类推直到字符串不能再切割
 */
func PalindromePartition(s string) [][]string {
	return palindromePartitionHelper(s, 0)
}

func palindromePartitionHelper(s string, start int) [][]string {
	ress := [][]string{}
	lens := len(s)
	if start == lens { // 递归出口为 idx 到达字符串末端
		res := []string{}
		ress = append(ress, res)
		return ress
	}
	for i := start; i < lens; i++ {
		left := s[start : i+1]
		if !IsPalindrome(left) { // 对于不是回文的子串就跳过了
			continue
		}
		// 处理子串
		for _, l := range palindromePartitionHelper(s, i+1) {
			ll := make([]string, len(l)+1)
			ll[0] = left
			copy(ll[1:], l) // 将 left 放在数组最前端
			ress = append(ress, ll)
		}
	}
	return ress
}

/*
 * 解法 2：分治法的优化
 * 判断字符串 abbbba 是否是回文串时，肯定会判断 bbbb 是不是回文串，
 * 其实如果我们已经知道了 bbbb 是回文串，只需要判断 abbbba 的开头和末尾字符是否相等即可。
 * 用 dp[i][j] 记录 s[i，j] 是否是回文串，可以避免很多不必要的比较，提前结束
 */
func PalindromePartition2(s string) [][]string {
	slen := len(s)
	dp := make([][]bool, slen)
	for i := range dp {
		dp[i] = make([]bool, slen)
	}
	for ln := 1; ln <= slen; ln++ {
		for i := 0; i <= slen-ln; i++ {
			j := i + ln - 1
			dp[i][j] = (s[i] == s[j]) && (ln < 3 || dp[i+1][j-1])
		}
	}
	return palindromePartitionHelper2(s, 0, dp)

}

func palindromePartitionHelper2(s string, start int, dp [][]bool) [][]string {
	ress := [][]string{}
	slen := len(s)
	if start == slen { // 递归出口为 idx 到达字符串末端
		res := []string{}
		ress = append(ress, res)
		return ress
	}
	for i := start; i < slen; i++ {
		if !dp[start][i] {
			continue
		}
		// 处理子串
		left := s[start : i+1]
		for _, l := range palindromePartitionHelper(s, i+1) {
			ll := make([]string, len(l)+1)
			ll[0] = left
			copy(ll[1:], l) // 将 left 放在数组最前端
			ress = append(ress, ll)
		}
	}
	return ress
}

/*
 * 解法 3：回溯法
 * DFS 深度优先遍历，套用回溯法的模板
 */
func PalindromePartition3(s string) [][]string {
	slen := len(s)
	dp := make([][]bool, slen)
	for i := range dp {
		dp[i] = make([]bool, slen)
	}
	for ln := 1; ln <= slen; ln++ {
		for i := 0; i <= slen-ln; i++ {
			j := i + ln - 1
			dp[i][j] = (s[i] == s[j]) && (ln < 3 || dp[i+1][j-1])
		}
	}
	var ress [][]string
	var temp []string
	palindromePartitionHelper3(s, 0, &ress, temp, dp)
	return ress
}

func palindromePartitionHelper3(s string, start int, ress *[][]string, path []string, dp [][]bool) {
	slen := len(s)
	if slen == start {
		tmp := make([]string, len(path))
		copy(tmp, path) // 深拷贝，防止上一层的删除影响到这一次的 slice
		*ress = append(*ress, tmp)
		return
	}
	for i := start; i < slen; i++ {
		if !dp[start][i] { // 判断start 到 i 子串是否是回文串
			continue
		}
		path = append(path, s[start:i+1])
		palindromePartitionHelper3(s, i+1, ress, path, dp)
		path = path[:len(path)-1] // 回溯，删掉上一次塞进来的 s[start:i+1]
	}
	return
}

/**
 * leetcode 题 344 反转字符串
 * 编写一个函数，其作用是将输入的字符串反转过来。
 * 说明：不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
 *
 * 输入：["h","e","l","l","o"]
 * 输出：["o","l","l","e","h"]
 *
 * https://leetcode-cn.com/problems/reverse-string/
 */
func ReverseString(s []byte) {
	l := 0
	r := len(s) - 1
	for l < r {
		s[l], s[r] = s[r], s[l]
		l++
		r--
	}
}

/**
 * leetcode T242. 有效的字母异位词
 * https://leetcode-cn.com/problems/valid-anagram/
 * 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词
 *
 * 输入: s = "anagram", t = "nagaram"
 * 输出: true
 *
 */
// 方法 1：哈希表法
// 使用哈希表更为通用，如果字符串仅仅是小写英文字母，也可以定义一个 26 长度的数组来统计字符频度
func IsAnagram(s string, t string) bool {
	if len(s) != len(t) { // 长度不一样的俩字符串肯定不是异位词
		return false
	}
	cnts := make(map[rune]int)
	// 也可以同时遍历 s 和 t
	for _, cr := range s {
		cnts[cr]++
	}
	for _, cr := range t {
		cnts[cr]--
		if cnts[cr] < 0 {
			return false
		}
	}
	return true
}

// 方法 2 排序法
// 将两个字节数组字母先排序，看他们是否相等

/*
 * 字符串中数字字串的求和
 * 给定一个字符串 str，求其中全部包含数字串所代表的数字之和
 * 要求：
 * 1. 忽略小数点
 * 2. 如果紧贴数字字串左侧出现字符 '-'，当连续出现的数量为奇数时，则视数字为负，否则为正。
 * 例如，A-1BC--12，计算结果 -1 + 12 = 11
 */
func NumSum(s string) int {
	if s == "" {
		return 0
	}
	sum := 0
	num := 0 // 累加字符串中出现的数字
	sign := 1
	for i, char := range s {
		curr := char - '0'
		if curr > 9 || curr < 0 { // 非数字字符
			sum += num // 在遇到非数字字符时累加 sum
			num = 0    // 累积数字清零
			if char != '-' {
				sign = 1 // 累积符号清空
				continue
			}
			if i != 0 && s[i-1] == '-' {
				sign = -sign
			} else {
				sign = -1 // 第一个字符是 - 号
			}
		} else {
			num = 10*num + int(curr)*sign // 在遇到数字字符时累加 num
		}
	}
	sum += num // 兜住字符串最后字符为数字的情况
	return sum
}
