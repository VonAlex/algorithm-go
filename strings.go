package leetcode

import (
	"container/list"
	"math"
	"strconv"
	"strings"
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

func RotateString(A string, B string) bool {
	if len(A) != len(B) {
		return false
	}
	BB := B + B
	return strings.Contains(BB, A)
}

/*
 * LeetCode T28. 实现 strStr()
 * https://leetcode-cn.com/problems/implement-strstr/
 *
 * 给定一个 haystack 字符串和一个 needle 字符串，
 * 在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
 *
 */

// 方法 1：BF(Brute Force，暴力检索)
// 时间复杂度 O(MN)
// i，j 都要回溯

// 方法 2：RK(Robin-Karp，哈希检索)

// 方法 3：KMP 算法
// 整个算法最坏的情况是，当模式串首字符位于i - j的位置时才匹配成功，算法结束。
// 如果文本串的长度为n，模式串的长度为m
// 匹配过程的时间复杂度为O(n)，算上计算 next的O(m)时间，KMP的整体时间复杂度为O(m + n)
func KmpSearch(text, pattern string) int {
	i := 0
	j := 0
	textLen := len(text)
	pLen := len(pattern)
	if pLen == 0 {
		return 0
	}
	next := getNext(pattern)
	for i < textLen && j < pLen {
		if j == -1 || text[i] == pattern[j] {
			i++
			j++
		} else {
			j = next[j] // j 的回溯，i 不回溯
		}
	}
	if j == pLen {
		return i - j
	}
	return -1
}

// 获得 next 数组
func getNext(p string) []int {
	pLen := len(p)
	next := make([]int, pLen)
	next[0] = -1
	k := -1
	j := 0
	for j < pLen-1 {
		// p[k]表示前缀，p[j]表示后缀
		if k == -1 || p[k] == p[j] {
			k++
			j++
			if p[k] != p[j] {
				next[j] = k // j+1
			} else {
				// 优化点
				// 不能出现p[j] = p[next[j]]，所以当出现时需要继续递归，k=next[k]=next[next[k]]
				next[j] = next[k]
			}
		} else {
			k = next[k]
		}
	}
	return next
}

// 方法 5：BM（Boyer Moore)

// 方法 6：Sunday 算法
// 思想跟BM算法很相似：
// 只不过Sunday算法是从前往后匹配，在匹配失败时关注的是主串中参加匹配的最末位字符的下一位字符。
// 如果该字符没有在模式串中出现则直接跳过，即移动位数 = 模式串长度 + 1；
// 否则，其移动位数 = 模式串长度 - 该字符最右出现的位置(以0开始)
// 时间复杂度，最差情况O（MN），最好情况O（N）
func SundaySearch(text, pattern string) int {
	pLen := len(pattern)
	if pLen == 0 {
		return 0
	}
	tLen := len(text)
	asciiSize := 128
	next := make([]int, asciiSize)
	for i := 0; i < asciiSize; i++ {
		next[i] = pLen + 1 // 初始值全部为 pLen + 1，当然这里也可以用 map 来做
	}
	for offset, c := range pattern {
		next[c] = pLen - offset
	}

	i := 0 // pattern 的头部元素在 text 中位置
	j := 0 // pattern 与 text 已经匹配的长度
	for i <= tLen-pLen {
		j = 0
		for text[i+j] == pattern[j] {
			j++
			if j >= pLen {
				return i
			}
		}
		// 最后一次匹配失败了，直接返回 -1
		if i+pLen == tLen {
			return -1
		}
		i += next[text[i+pLen]]
	}
	return -1
}

/*
 * LeetCode 面试题58 - II. 左旋转字符串
 * https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/
 *
 * 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。
 * 请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。
 */

// 方法 1：切片法
// trick 方法，并不是题目想考察的点
func ReverseLeftWords(s string, n int) string {
	if n == 0 {
		return s
	}
	res := s[n:] + s[:n]
	return res
}

// 方法 2：三次旋转法
// 时间复杂度为O(n)
func ReverseLeftWords2(s string, n int) string {
	if n == 0 {
		return s
	}
	_reverStr := func(sbytes []byte, from, to int) {
		for from <= to {
			sbytes[from], sbytes[to] = sbytes[to], sbytes[from]
			from++
			to--
		}
	}
	slen := len(s)
	sbytes := []byte(s)
	_reverStr(sbytes, 0, n-1)
	_reverStr(sbytes, n, slen-1)
	_reverStr(sbytes, 0, slen-1)
	return string(sbytes)
}

/*
 * LeetCode T151. 翻转字符串里的单词
 * https://leetcode-cn.com/problems/reverse-words-in-a-string/
 *
 * 给定一个字符串，逐个翻转字符串中的每个单词。
 * 示例：
 * 输入: "  hello world!  "
 * 输出: "world! hello"
 * 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
 *
 * 输入: "a good   example"
 * 输出: "example good a"
 * 解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
 */
// 方法 1
// golang 字符串无法修改，所以需要额外空间存储
// string 与 []byte 进行类型转换的时候，涉及到内存拷贝
func ReverseWords(s string) string {
	if s == "" {
		return s
	}
	// 去掉冗余空格
	_trimSpace := func(s string) string {
		l := 0
		r := len(s) - 1

		// 去掉字符串开头的空白字符
		for l <= r && s[l] == ' ' {
			l++
		}
		// 去掉字符串末尾的空白字符
		for l <= r && s[r] == ' ' {
			r--
		}
		// 将字符串间多余的空白字符去除
		var output []byte
		for l <= r {
			if s[l] == ' ' && s[l-1] == ' ' {
				l++
				continue
			}
			output = append(output, s[l])
			l++
		}
		return string(output)
	}
	// 反转单词
	_reverStr := func(sbytes []byte, from, to int) {
		for from <= to {
			sbytes[from], sbytes[to] = sbytes[to], sbytes[from]
			from++
			to--
		}
	}

	sb := []byte(_trimSpace(s))
	wordStart := 0 // 记录单词初始位置
	curr := 0      // 记录单词结束位置
	slen := len(sb)
	for wordStart < slen {
		for curr < slen && sb[curr] != ' ' {
			curr++
		}
		_reverStr(sb, wordStart, curr-1)
		wordStart = curr + 1 // wordEnd 停在了空格上，下一个单词
		curr = wordStart
	}

	_reverStr(sb, 0, slen-1)
	return string(sb)
}

// 方法 2：使用strings 包内的函数
func ReverseWords2(s string) string {
	if s == "" {
		return s
	}
	s = strings.TrimSpace(s)
	res := strings.Fields(s)
	i := 0
	j := len(res) - 1
	for i <= j {
		res[i], res[j] = res[j], res[i]
		i++
		j--
	}
	return strings.Join(res, " ")
}

// 方法 3：借助栈结构
func ReverseWords3(s string) string {
	if s == "" {
		return s
	}
	l := 0
	r := len(s) - 1
	for l <= r && s[l] == ' ' {
		l++
	}
	for l <= r && s[r] == ' ' {
		r--
	}
	stack := list.New()
	wordStart := l
	curr := l
	for curr <= r {
		if s[curr] != ' ' {
			curr++
			continue
		}
		stack.PushFront(string(s[wordStart:curr]))
		for curr <= r && s[curr] == ' ' {
			curr++
		}
		wordStart = curr
	}
	if wordStart <= r {
		stack.PushFront(string(s[wordStart:curr]))
	}

	var res []string
	for e := stack.Front(); e != nil; e = e.Next() {
		res = append(res, e.Value.(string))
	}
	return strings.Join(res, " ")
}

// 输出 s 中的单词列表
func SplitWords(s string) []string {
	if s == "" {
		return nil
	}
	sLen := len(s)
	curr := 0
	wordStart := curr
	var res []string
	for curr < sLen {
		if s[curr] != ' ' {
			curr++
			continue
		}
		res = append(res, s[wordStart:curr])
		curr++
		for curr < sLen && s[curr] == ' ' {
			curr++
		}
		wordStart = curr
	}
	if wordStart < sLen {
		res = append(res, s[wordStart:curr])
	}
	return res
}

// 计算 s 中有多少了单词(忽略掉空格)
func CountFields(s string) int {
	n := 0
	var isSpace int
	wasSpace := 1
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			isSpace = 1
		} else {
			isSpace = 0
		}
		n += wasSpace & ^isSpace
		wasSpace = isSpace
	}
	return n
}
