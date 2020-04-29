package leetcode

import (
	"container/list"
	"unicode"
)

/*********************************** 对撞指针题目 ********************************************/

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
// 双指针法（对撞指针）
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

/**
 * leetcode T344. 反转字符串
 * https://leetcode-cn.com/problems/reverse-string/
 *
 * 编写一个函数，其作用是将输入的字符串反转过来。
 * 说明：不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
 *
 * 输入：["h","e","l","l","o"]
 * 输出：["o","l","l","e","h"]
 */
// 双指针法（对撞指针）
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
 * leetcode T345. 反转字符串中的元音字母
 * https://leetcode-cn.com/problems/reverse-vowels-of-a-string/
 *
 * 编写一个函数，以字符串作为输入，反转该字符串中的元音字母。
 * 说明：不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
 * 示例：
 * 输入："hello"
 * 输出： "holle"
 *
 * 说明:元音字母不包含字母"y"。
 */
// 注意：题目中没有告诉元音字符是大小写
// 方法 1：双指针法（对撞指针）
func ReverseVowels(s string) string {
	if s == "" {
		return s
	}
	vowels := map[byte]bool{
		'a': true,
		'e': true,
		'i': true,
		'o': true,
		'u': true,
		'A': true,
		'E': true,
		'I': true,
		'O': true,
		'U': true,
	}
	l := 0
	r := len(s) - 1
	bts := []byte(s)
	for l < r {
		_, ok := vowels[bts[l]]
		if !ok {
			l++
			continue
		}
		_, ok = vowels[bts[r]]
		if !ok {
			r--
			continue
		}
		bts[l], bts[r] = bts[r], bts[l]
		l++
		r--
	}
	return string(bts)
}

// 方法 2：利用栈结构
func ReverseVowels2(s string) string {
	if s == "" {
		return s
	}
	vowels := map[byte]bool{
		'a': true,
		'e': true,
		'i': true,
		'o': true,
		'u': true,
		'A': true,
		'E': true,
		'I': true,
		'O': true,
		'U': true,
	}
	vowelsIns := list.New() // 栈存放 s 中出现的原因字母
	sLen := len(s)
	for i := 0; i < sLen; i++ {
		_, ok := vowels[s[i]]
		if ok {
			vowelsIns.PushBack(s[i])
		}
	}
	bts := []byte(s)
	for i := 0; i < sLen; i++ {
		_, ok := vowels[s[i]]
		if ok {
			// 遍历 s 遇到元音字母，从栈中 pop 一个进行替换
			bts[i] = vowelsIns.Remove(vowelsIns.Back()).(byte)
		}
	}
	return string(bts)
}

/**
 * leetcode T11. 盛最多水的容器
 * https://leetcode-cn.com/problems/container-with-most-water/
 *
 * 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。
 * 在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。
 * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
 *
 * 说明：你不能倾斜容器，且 n 的值至少为 2。
 * 示例：
 * 输入：[1,8,6,2,5,4,8,3,7]
 * 输出：49
 */
// 双指针法（对撞指针）

// 水槽面积 S(i, j) = min(h[i], h[j]) × (j - i)
// 在每一个状态下，无论长板或短板收窄 1 格，都会导致水槽底边宽度 -1
// 若向内移动短板，水槽的短板 min(h[i],h[j]) 可能变大，因此水槽面积 S(i, j)可能增大。
// 若向内移动长板，水槽的短板 min(h[i],h[j]) 不变或变小，下个水槽的面积一定小于当前水槽面积
func MaxArea(height []int) int {
	l := 0
	r := len(height) - 1
	maxArea := 0
	for l < r {
		// 收窄短板
		if height[l] < height[r] {
			area := (r - l) * height[l]
			if area > maxArea {
				maxArea = area
			}
			l++
		} else {
			area := (r - l) * height[r]
			if area > maxArea {
				maxArea = area
			}
			r--
		}
	}
	return maxArea
}

/**
 * LeetCode T167. 两数之和 II - 输入有序数组
 * 给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数
 * 函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
 * 说明:
 *  	返回的下标值（index1 和 index2）不是从零开始的。
 *		你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
 */
// 方法 1：双指针法
// 时间复杂度：O(n)，空间复杂度：O(1)
// 要注意没有合适答案时删除什么，这个要做好约定
func TwoSum4(numbers []int, target int) []int {
	i := 0
	j := len(numbers) - 1
	for i <= j {
		if numbers[i]+numbers[j] == target {
			return []int{i + 1, j + 1}
		} else if numbers[i]+numbers[j] > target {
			j--
		} else {
			i++
		}
	}
	return []int{-1, -1} // 找到不到 target 返回什么
}

// 方法 2：双指针 + 二分查找
// 时间复杂度 O(nlogn)，n 是指数组长度
// 设二分查找每次过滤掉 d 个数，如果 d 较小时，可能二分法不怎么占优势
func TwoSum5(numbers []int, target int) []int {
	i := 0
	j := len(numbers) - 1

	_bs := func(numbers []int, l, r, t int) int {
		var mid int
		for l <= r {
			mid = (l + r) >> 1
			if numbers[mid] == t {
				break
			} else if numbers[mid] > t {
				r = mid - 1
			} else {
				l = mid + 1
			}
		}
		return mid
	}
	for i <= j {
		if numbers[i]+numbers[j] == target {
			return []int{i + 1, j + 1}
		} else if numbers[i]+numbers[j] > target {
			// 从 [i, j-1] 重新定位 j
			j = _bs(numbers, i, j-1, target-numbers[i])
		} else {
			// 从 [i+1, j] 重新定位 i
			i = _bs(numbers, i+1, j, target-numbers[j])
		}
	}
	return []int{-1, -1}
}

/*************************************** 滑动窗口题目(前后指针) *******************************************/
/*
 * LeetCode T76. 最小覆盖子串
 * https://leetcode-cn.com/problems/minimum-window-substring/
 *
 * 给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串。
 * 示例：
 * 输入: S = "ADOBECODEBANC", T = "ABC"
 * 输出: "BANC"
 * 说明：
 * 		如果 S 中不存这样的子串，则返回空字符串 ""
 * 		如果 S 中存在这样的子串，我们保证它是唯一的答案。
 *
 */
// 滑动窗口解法
// S 的长度为 M，T 的长度为 N，时间复杂度为 O(M+N)
// 遍历 T 的时间复杂度 O(N)
// 2 个 while 循环里最多执行 2M 次，时间复杂度为 O(M)

// 另外，可以如下优化滑动窗口，将 T 中不存在的字符从 S 中去掉
// S 变成一个 map, key 是原 S 中字符的位置，value 是字符
// 下面遍历 S 就变成了遍历这个 map
// 因为 go 中 map 遍历是无序的，所有要把 key 用一个 list，有序存起来
// 按照这个有序 list 进行遍历 map
func MinWindow(s string, t string) string {
	if s == "" || t == "" {
		return ""
	}
	left := 0                    // 窗口左侧
	right := 0                   // 窗口右侧
	window := make(map[byte]int) // 窗口

	start := 0   // 最小子串开始的位置
	minLen := -1 // 最小子串的长度

	needs := make(map[byte]int) // 需要匹配的字符计数器
	tLen := len(t)
	for i := 0; i < tLen; i++ {
		needs[t[i]]++
	}
	needsMacth := len(needs)
	matched := 0 // 窗口中已经匹配到字符数

	slen := len(s)
	for right < slen { // 遍历 s
		c1 := s[right] // 窗口右侧
		if _, ok := needs[c1]; ok {
			window[c1]++                 // c1 在窗口中出现的次数
			if window[c1] == needs[c1] { // t 中的 c1 字符被 window 全部包含
				matched++
			}
		}

		for matched == needsMacth { // 窗口中已经包含了 t 中的所有字符
			newWindowSize := right - left + 1
			if minLen == -1 || newWindowSize < minLen { // 更新窗口大小
				start = left
				minLen = newWindowSize
			}
			c2 := s[left] // 窗口左侧
			if _, ok := needs[c2]; ok {
				window[c2]--
				if window[c2] < needs[c2] {
					matched-- // 失配一个字符
				}
			}
			left++ // 窗口收紧左边侧
		}

		right++ // 窗口右移
	}
	if minLen == -1 {
		return ""
	}
	return s[start : start+minLen]
}

/*
 * LeetCode T438. 找到字符串中所有字母异位词
 * https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/
 *
 * 给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。
 * 字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100
 *
 * 示例：
 * 输入: s: "cbaebabacd" p: "abc"
 * 输出: [0, 6]
 *
 * 说明：
 * 		字母异位词指字母相同，但排列不同的字符串。
 * 		不考虑答案输出的顺序。
 *
 */
// 滑动窗口解法
// 与“最小覆盖子串”题解基本上是相同的，只是结果输出需要变一下
// 上一题是求解最小窗口，所以需要 start 和 minLen 去记录历史值，方面比较
// 本题长度就等于 p 字符串的长度，而且结果保存到 slice 里
// 因为，异位词的长度肯定要跟窗口长度相等
func FindAnagrams(s string, p string) []int {
	left := 0
	right := 0
	window := make(map[byte]int)

	pLen := len(p)
	needs := make(map[byte]int)
	for i := 0; i < pLen; i++ {
		needs[p[i]]++
	}
	needMatched := len(needs)
	matched := 0

	var res []int
	sLen := len(s)
	for right < sLen {
		c1 := s[right]
		if _, ok := needs[c1]; ok {
			window[c1]++
			if window[c1] == needs[c1] {
				matched++
			}
		}
		for matched == needMatched {
			if right-left+1 == pLen { // 长度等于 t 的长度
				res = append(res, left)
			}
			c2 := s[left]
			if _, ok := needs[c2]; ok {
				window[c2]--
				if window[c2] < needs[c2] {
					matched--
				}
			}
			left++
		}
		right++
	}
	return res
}

/*
 * LeetCode T3. 无重复字符的最长子串
 * https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
 *
 * 给定一个字符串，请你找出其中不含有重复字符的最长子串的长度
 *
 * 示例：
 * 输入: "abcabcbb"
 * 输出: 3
 * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
 */
// 滑动窗口法
func LengthOfLongestSubstring(s string) int {
	left := 0
	right := 0
	window := make(map[byte]int)
	sLen := len(s)
	res := 0
	for right < sLen {
		c1 := s[right]
		window[c1]++

		for window[c1] > 1 {
			c2 := s[left]
			window[c2]--
			left++
		}
		diff := right - left + 1
		if res < diff {
			res = diff
		}
		right++
	}
	return res
}

/*
 * 滑动窗口小结
 * left := 0
 * right := 0
 * for right < len(s) {
 *     c1 := s[right]
 *     window[c1]++
 *     for 条件 {
 *	       c2 := s[left]
 *         window[c2]--
 *         // 在再次移动窗口左侧之前，可能有条件处理
 *         left++
 *     }
 *     // 在再次右移动窗口右侧之前，可能有条件处理，
 *     right++
 * }
 */

/**
 * LeetCode T209. 长度最小的子数组
 * https://leetcode-cn.com/problems/minimum-size-subarray-sum/
 *
 * 给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组。
 * 如果不存在符合条件的连续子数组，返回 0。
 *
 * 示例：
 * 输入: s = 7, nums = [2,3,1,2,4,3]
 * 输出: 2
 * 解释: 子数组 [4,3] 是该条件下的长度最小的连续子数组。
 */
// 滑动窗口法
// 时间复杂度 O(n)
func MinSubArrayLen(s int, nums []int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
	}

	var l, r, sum, minLen int
	for r < numsLen {
		sum += nums[r]
		for sum >= s {
			if minLen == 0 || minLen > r-l+1 {
				minLen = r - l + 1
			}
			sum -= nums[l]
			l++ // 窗口左侧移动
		}
		r++ //窗口右侧移动
	}
	return minLen
}

// 循环+二分
// 时间复杂度 O(nlogn)
func MinSubArrayLen2(s int, nums []int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
	}
	sumsLen := numsLen + 1
	sums := make([]int, sumsLen)

	// 积和数组
	// sum[0] = 0
	// sum[i] 表示 num 数组中 0 ~ i-1 数字之和
	for i := 1; i < sumsLen; i++ {
		sums[i] = nums[i-1] + sums[i-1]
	}
	minLen := 0
	for i := 0; i <= numsLen; i++ { // 循环 O(n)
		l := i
		r := sumsLen - 1
		for l <= r { // 二分 O(logn)
			mid := l + (r-l)>>1
			subSum := sums[mid] - sums[i] // i ~ mid-1 的数值累积和
			if subSum >= s {
				r = mid - 1
				if minLen == 0 || minLen > mid-i {
					minLen = mid - i
				}
			} else {
				l = mid + 1 // 即，mid 左边数之和小于 s
			}
		}
	}
	return minLen
}

/*************************************** 二分查找题目 *******************************************/

/*
 * LeetCode T1704. 二分查找
 * https://leetcode-cn.com/problems/binary-search/
 * 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target ，
 * 写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
 */
func BinarySearch(nums []int, target int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return -1
	}
	l := 0
	r := numsLen - 1
	for l <= r {
		mid := l + (r-l)>>1
		if target == nums[mid] {
			return mid
		} else if target > nums[mid] {
			l = mid + 1
		} else if target < nums[mid] {
			r = mid - 1
		}
	}
	return -1
}

// 二分查找，寻找数组中相同元素的左边界
// [1, 2, 2, 2, 3] -> 1
func BinarySearchFindLeftBound(nums []int, target int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return -1
	}
	l := 0
	r := numsLen - 1
	for l <= r {
		mid := l + (r-l)>>1
		if target == nums[mid] {
			r = mid - 1
		} else if target > nums[mid] {
			l = mid + 1
		} else if target < nums[mid] {
			r = mid - 1
		}
	}
	// 异常判断
	if l >= numsLen || nums[l] != target {
		return -1
	}
	return l
}

// 解题思路可以参考 https://leetcode-cn.com/problems/binary-search/solution/er-fen-cha-zhao-xiang-jie-by-labuladong/
// 二分查找，寻找数组中相同元素的右边界
// [1, 2, 2, 2, 3] -> 3
func BinarySearchFindRightBound(nums []int, target int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return -1
	}
	l := 0
	r := numsLen - 1
	for l <= r {
		mid := l + (r-l)>>1
		if target == nums[mid] {
			l = mid + 1
		} else if target > nums[mid] {
			l = mid + 1
		} else if target < nums[mid] {
			r = mid - 1
		}
	}
	// 异常判断
	if r < 0 || nums[r] != target {
		return -1
	}
	return r
}
