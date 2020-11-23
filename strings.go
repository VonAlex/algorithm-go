package leetcode

import (
	"container/list"
	"strconv"
	"strings"
)

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

/* leetcode Ts387. 字符串中的第一个唯一字符
 * https://leetcode-cn.com/problems/first-unique-character-in-a-string/
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

/****************************************************************************************/

/*
 * leetcode T131 分割回文串
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
 * LeetCode 面试题58 - I. 翻转单词顺序
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
// 方法 1 三次旋转法
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

// 计算 s 中有多少单词(忽略掉空格)
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

/*
 * LeetCode T67. 二进制求和
 * https://leetcode-cn.com/problems/add-binary/
 *
 * 给给你两个二进制字符串，返回它们的和（用二进制表示）。
 * 输入为 非空 字符串且只包含数字 1 和 0。
 * 示例：
 * 输入: a = "11", b = "1"
 * 输出: "100"
 */
// 方法 1：逐位相加
func addBinary(a string, b string) string {
	if a == "" {
		return b
	}
	if b == "" {
		return a
	}
	sum := ""
	alen, blen := len(a), len(b)
	maxlen := alen
	if blen > maxlen {
		maxlen = blen
	}
	carry := 0
	for i := 0; i < maxlen; i++ {
		if i < alen {
			carry += int(a[alen-i-1] - '0')
		}
		if i < blen {
			carry += int(b[blen-i-1] - '0')
		}
		sum = strconv.Itoa(carry%2) + sum
		carry /= 2
	}
	if carry > 0 {
		sum = "1" + sum
	}
	return sum
}

// 方法 2： 位运算
// 计算 x 和 y 的无进位相加结果：answer = x ^ y
// 计算 x 和 y 的进位：carry = (x & y) << 1
// 在第一轮计算中，answer 的最后一位是 x 和 y 相加之后的结果，carry 的倒数第二位是 x 和 y 最后一位相加的进位。
// 接着每一轮中，由于 carry 是由 x 和 y 按位与并且左移得到的，那么最后会补零，所以在下面计算的过程中后面的数位不受影响，
// s而每一轮都可以得到一个低 i 位的答案和它向低 i+1 位的进位，也就模拟了加法的过程。
func addBinary2(a string, b string) string {
	if a == "" {
		return b
	}
	if b == "" {
		return a
	}
	ai, _ := strconv.ParseInt(a, 2, 10)
	bi, _ := strconv.ParseInt(b, 2, 10)
	for bi != 0 {
		carry := ai & bi
		ai ^= bi
		bi = carry << 1
	}
	return strconv.FormatInt(ai, 2)
}

/*
 * LeetCode T844. 比较含退格的字符串
 * https://leetcode-cn.com/problems/backspace-string-compare/
 *
 * 给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。
 * 注意：如果对空文本输入退格字符，文本继续为空。
 *
 * 示例：
 * 输入：S = "a##c", T = "#a#c"
 * 输出：true
 * 解释：S 和 T 都会变成 “c”.
 */
// 方法 1：辅助栈法
// 时间/空间复杂度：O(N+M)
func backspaceCompare(S string, T string) bool {
	stackS := list.New()
	for _, c := range S {
		if c == '#' {
			if stackS.Len() > 0 {
				stackS.Remove(stackS.Back())
			}
			continue
		}
		stackS.PushBack(c)
	}

	stackT := list.New()
	for _, c := range T {
		if c == '#' {
			if stackT.Len() > 0 {
				stackT.Remove(stackT.Back())
			}
			continue
		}
		stackT.PushBack(c)
	}
	if stackS.Len() != stackT.Len() {
		return false
	}
	for stackS.Len() > 0 && stackT.Len() > 0 {
		s := stackS.Remove(stackS.Back()).(byte)
		t := stackT.Remove(stackT.Back()).(byte)
		if s != t {
			return false
		}
	}
	return true
}

// 方法 2：辅助数组法
// 时间/空间复杂度：O(N+M)
func backspaceCompare2(S string, T string) bool {
	_rebuild := func(s string) string {
		var res []byte
		for i := range s {
			if s[i] != '#' {
				res = append(res, s[i])
			} else if len(res) > 0 {
				res = res[:len(res)-1]
			}
		}
		return string(res)
	}
	return _rebuild(S) == _rebuild(T)
}

// 方法 3：双指针法
func backspaceCompare3(S string, T string) bool {
	skipS, skipT := 0, 0 // skip 表示需要退格几次
	i, j := len(S)-1, len(T)-1
	for i >= 0 || j >= 0 {
		for i >= 0 {
			if S[i] == '#' {
				skipS++
				i--
			} else if skipS > 0 { // 删掉一个字符
				skipS--
				i--
			} else { // 当前字符不需要删除
				break
			}
		}
		for j >= 0 {
			if T[j] == '#' {
				skipT++
				j--
			} else if skipT > 0 {
				skipT--
				j--
			} else {
				break
			}
		}
		if i >= 0 && j >= 0 {
			if S[i] != T[j] { // 不需要退格时的字符是否相等
				return false
			}
		} else if i >= 0 || j >= 0 { // 有一个遍历完了，而另一个没有
			return false
		}
		i--
		j--
	}
	return true
}

/*
 * LeetCode T14. 最长公共前缀
 * https://leetcode-cn.com/problems/backspace-string-compare/
 *
 * 编写一个函数来查找字符串数组中的最长公共前缀。
 * 如果不存在公共前缀，返回空字符串 ""。
 *
 * 示例：
 * 输入：["flower","flow","flight"]
 * 输出："fl"
 *
 * 说明：所有输入只包含小写字母 a-z 。
 */
// 方法 1：纵向扫描
// 纵向扫描时，从前往后遍历所有字符串的每一列，比较相同列上的字符是否相同，
// 如果相同则继续对下一列进行比较，
// 如果不相同则当前列不再属于公共前缀，当前列之前的部分为最长公共前缀。
//
// 时间复杂度：O(mn)，空间复杂度：O(1)
func longestCommonPrefix(strs []string) string {
	arrLen := len(strs)
	if arrLen == 0 {
		return ""
	}
	slen := len(strs[0])
	for i := 0; i < slen; i++ {
		for j := 0; j < arrLen; j++ {
			// 只有所有字符串第 i 个字符相等才会继续比较
			if i >= len(strs[j]) || strs[0][i] != strs[j][i] {
				return strs[0][:i]
			}
		}
	}
	return strs[0] // strs[0] 就是最长前缀
}

// 方法 2：横向扫描
// 设 LCP(S1..Sn) 表示字符串 S1..Sn 的最长公共前缀，
// 那么 LCP(S1..Sn) = LCP(LCP(LCP(S1,S2),S3)..Sn)
// 依次遍历每个字符串，更新最长公共前缀。
func longestCommonPrefix2(strs []string) string {
	arrLen := len(strs)
	if arrLen == 0 {
		return ""
	}
	prefix := strs[0]
	for i := 1; i < arrLen; i++ {
		prefix = lcp(prefix, strs[i])
		// 最长公共前缀是空串，则整个数组字符串的最长公共前缀一定是空串
		if prefix == "" {
			return ""
		}
	}
	return prefix
}

func lcp(str1, str2 string) string {
	len1, len2 := len(str1), len(str2)
	var i int
	for i < len1 && i < len2 {
		if str1[i] != str2[i] {
			break
		}
		i++
	}
	return str1[:i]
}
