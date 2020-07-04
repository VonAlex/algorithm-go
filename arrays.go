package leetcode

import (
	"math"
	"math/rand"
	"sort"
)

/**
 * LeetCode T1 两数之和
 * 给定一个整数数组 nums 和一个目标值 target，
 * 请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。
 */
// 这里的数组应该是不含相同元素的数组
// TwoSum 暴力法 Brute Force
// 时间复杂度 O(n^2)，空间复杂度 O(1)
func TwoSum(nums []int, target int) []int {
	len := len(nums)
	for i, num := range nums {
		for j := i + 1; j < len; j++ {
			if num+nums[j] != target {
				continue
			}
			return []int{i, j}
		}
	}
	return []int{-1, -1}
}

// TwoSum2 两遍 hash 表法
// 时间复杂度 O(n)，空间复杂度 O(n)
func TwoSum2(nums []int, target int) []int {
	numIdx := make(map[int]int)
	// 如果说数组中有相同的数字，那么使用 hash 是会覆盖掉相同元素
	for idx, num := range nums {
		numIdx[num] = idx
	}

	for i, num := range nums {
		j, ok := numIdx[target-num]
		// 防止索引到自身，如 6=3+3
		// 如果是数组中是 3，但是给出的 taget 是 6，这样是有问题的
		if ok && i != numIdx[j] {
			return []int{i, j}
		}
	}
	return nil
}

// TwoSum3 一遍 hash 表法
// 时间复杂度 O(n)，空间复杂度 O(n)
// 由于相同元素的覆盖，一遍遍历 hash 表和两遍遍历，获得的结果是不同的
func TwoSum3(nums []int, target int) []int {
	numIdx := make(map[int]int)
	for i, num := range nums {
		j, ok := numIdx[target-num]
		if !ok {
			numIdx[num] = i
			continue
		}
		return []int{i, j}
	}
	return nil
}

/**
 * LeetCode T189. 旋转数组
 * https://leetcode-cn.com/problems/rotate-array/
 * 给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。
 *
 * 示例 1:
 * 输入: [1,2,3,4,5,6,7] 和 k = 3
 * 输出: [5,6,7,1,2,3,4]
 *
 * 解释:
 * 向右旋转 1 步: [7,1,2,3,4,5,6]
 * 向右旋转 2 步: [6,7,1,2,3,4,5]
 * 向右旋转 3 步: [5,6,7,1,2,3,4]
 */
// 方法1：使用额外的数组
// 时/空间复杂度都是 O(n)
// 原本数组里下标为 i 的，我们把它放到 (i+k)%数组长度的位置
func rotate(nums []int, k int) {
	if k <= 0 {
		return
	}
	lens := len(nums)
	res := make([]int, lens, lens)
	for i, num := range nums {
		res[(i+k)%lens] = num
	}
	for j, num := range res {
		nums[j] = num
	}
	return
}

// 方法2：反转法
// 时间复杂度 O(n)，空间复杂度 O(1)
func rotate2(nums []int, k int) {
	if k <= 0 {
		return
	}
	r := len(nums)
	k %= r
	l := 0
	reverseArr(nums, l, r-1) // 整体翻转
	reverseArr(nums, l, k-1) // 前 k 个数翻转
	reverseArr(nums, k, r-1) // 后面的数进行翻转
	return
}

func reverseArr(nums []int, l, r int) {
	for l < r {
		nums[l], nums[r] = nums[r], nums[l]
		l++
		r--
	}
	return
}

/**
 * LeetCode T283. 移动零
 * https://leetcode-cn.com/problems/move-zeroes/
 * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
 * 示例:
 * 		输入: [0,1,0,3,12]
 * 		输出: [1,3,12,0,0]
 * 说明:
 * 必须在原数组上操作，不能拷贝额外的数组。
 * 尽量减少操作次数。
 */
// 方法 1：双指针法1
// 先找出移动后，非零元素的右边界，然后在剩余位置填 0
// 时间复杂度 O(n)，需要做数组写入的次数为数组长度
func MoveZeroes(nums []int) {
	numsLen := len(nums)
	if numsLen == 0 {
		return
	}
	var l int                      // 左指针记录最后一个非 0 位置的下一位
	for r := 0; r < numsLen; r++ { // 后指针用来遍历
		if nums[r] != 0 {
			nums[l] = nums[r]
			l++
		}
	}
	for l < numsLen { // 后面的填充 0
		nums[l] = 0
		l++
	}
	return
}

// 方法 2：双指针法2
// 0 元素比较多时效果比上一个 solution 要好
func MoveZeroes2(nums []int) {
	numsLen := len(nums)
	if numsLen == 0 {
		return
	}
	var l, r int
	for r < numsLen {
		if nums[r] != 0 {
			nums[l], nums[r] = nums[r], nums[l]
			l++
		}
		r++
	}
	return
}

/**
 * LeetCode T27. 移除元素
 * https://leetcode-cn.com/problems/remove-element/
 * 给你一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，并返回移除后数组的新长度。
 * 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。
 * 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
 * 示例:
 * 		给定 nums = [3,2,2,3], val = 3,
 * 		函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
 *      你不需要考虑数组中超出新长度后面的元素。
 */
// 解题思路跟上面的 “LeetCode T283. 移动零“ 完全一样
// 方法 1：双指针 —— 当要删除的元素元素较多时
func RemoveElement(nums []int, val int) int {
	numsLen := len(nums)
	var l, r int // r 从 0 开始是因为第一个元素也可能等于 val，不同于 26 题中的有序数组
	for r < numsLen {
		if nums[r] != val {
			// 交换，或者赋值都可以
			// nums[l], nums[r] = nums[r], nums[l]
			nums[l] = nums[r]
			l++
		}
		r++
	}
	return l // 最终， 0 ~ l-1 内的元素是想要的，长度为 l
}

// 方法 2：双指针 —— 当要删除的元素很少时
// 如果 nums 数组为 [1, 2, 3, 5, 4], val = 4
// 那么按照前面的方法会对前 4 个元素进行不必要的操作
func RemoveElement2(nums []int, val int) int {
	numsLen := len(nums)
	l := 0
	r := numsLen
	for l < r {
		if nums[l] == val {
			nums[l] = nums[r-1]
			r--
		} else {
			l++
		}
	}
	return l
}

/*
 * LeetCode T26. 删除排序数组中的重复项
 * https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/
 *
 * 给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
 * 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
 * 示例:
 * 		给定数组 nums = [1,1,2],
 * 		函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。
 *      你不需要考虑数组中超出新长度后面的元素。
 */
func RemoveDuplicates(nums []int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
	}
	var l, r int
	for r < numsLen {
		if nums[l] != nums[r] {
			l++ // 左指针先往前移动一个，错开上一次处理过的
			nums[l] = nums[r]
		}
		r++
	}
	return l + 1 // l 在被覆盖时，先加 1，不同于 25 题后加 1，所以最终， 0 ~ l 内的元素是想要的，长度为 l+1
}

/**
 * LeetCode T80. 删除排序数组中的重复项 II
 * https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/
 *
 * 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
 * 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
 * 示例:
 * 		给定数组  nums = [1,1,1,2,2,3],
 * 		函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3。
 *      你不需要考虑数组中超出新长度后面的元素。
 */
func RemoveDuplicates2(nums []int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
	}
	l := 2
	r := 2 // 前两个元素不能动
	for r < numsLen {
		if nums[r] != nums[l-2] {
			nums[l] = nums[r]
			l++
		}
		r++
	}
	return l
}

/**
 * LeetCode T217. 存在重复元素
 * https://leetcode-cn.com/problems/contains-duplicate/
 *
 * 给定一个整数数组，判断是否存在重复元素。
 * 如果任意一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。
 * 示例:
 * 		输入: [1,2,3,1]
 *      输出: true
 */

// 方法1：暴力法
func containsDuplicate(nums []int) bool {
	lens := len(nums)
	if lens == 0 {
		return false
	}

	for i := 0; i < lens-1; i++ {
		for j := i + 1; j < lens; j++ {
			if nums[i] == nums[j] {
				return true
			}
		}
	}
	return false
}

// 方法2：排序法
func containsDuplicate2(nums []int) bool {
	lens := len(nums)
	if lens == 0 {
		return false
	}

	sort.SliceStable(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})

	for i := 0; i < lens-1; i++ {
		if nums[i] == nums[i+1] {
			return true
		}
	}
	return false
}

// 方法3：哈希表法
func containsDuplicate3(nums []int) bool {
	lens := len(nums)
	if lens == 0 {
		return false
	}

	numSet := make(map[int]struct{})
	for _, num := range nums {
		if _, ok := numSet[num]; ok {
			return true
		}
		numSet[num] = struct{}{}
	}
	return false
}

/**
 * 剑指 offer 面试题03 找出数组中重复的数字
 * 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。
 * https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/
 *
 * 输入：[2, 3, 1, 0, 2, 5, 3]
 * 输出：2 或 3
 */

// FindRepeatNumber 利用 map 结构
// 时间复杂度和空间复杂度 O(n)
func FindRepeatNumber(nums []int) int {
	if len(nums) == 0 {
		return -1
	}
	sets := make(map[int]struct{})
	for _, num := range nums {
		if _, ok := sets[num]; ok {
			return num
		}
		sets[num] = struct{}{}
	}
	return -1
}

// FindRepeatNumber2 强调了长度为 n 的数组内数字范围都是 0-n-1。
// 可以想到，当数组内的数字经过排序后，那么下标 i，就应该等于数字 num，第一个不相等的，就是题解。
// 可以先排序，时间复杂度 O(nlogn)，再求解
// 也可以通过交换数字达到目的。时间复杂度 O(n),空间复杂度 O(1)
func FindRepeatNumber2(nums []int) int {
	if len(nums) == 0 {
		return -1
	}
	for i, num := range nums {
		if i == num { // 下标 == 数字，表示该数字已经在最早位置上了
			continue
		}
		if num == nums[num] { // 数字 num 排序后应该在序号为 num 的位置，但是这个位置上已经有一个数字了，等于 num，那么这就是题解
			return num
		}
		nums[i], nums[num] = nums[num], nums[i]
	}
	return -1
}

/**
 * 面试题 10.01 合并排序的数组
 * https://leetcode-cn.com/problems/sorted-merge-lcci/
 *
 * 输入:
 * 		A = [1,2,3,0,0,0], m = 3
 * 		B = [2,5,6],       n = 3
 *
 * 输出: [1,2,2,3,5,6]
 */

/*
 * LeetCode T88. 合并两个有序数组
 * https://leetcode-cn.com/problems/merge-sorted-array/
 * 面试题 10.01. 合并排序的数组
 * https://leetcode-cn.com/problems/sorted-merge-lcci/
 *
 * 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
 *
 * 说明:
 * 初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。
 * 你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
 *
 * 示例：
 *
 * 输入:
 * 		nums1 = [1,2,3,0,0,0], m = 3
 *		nums2 = [2,5,6],       n = 3
 *
 * 输出: [1,2,2,3,5,6]
 */
// 此题与合并排序链表是一样的，只是换成了数组
// 方法 1：双指针法，从后向前
// 时间复杂度 : O(n + m)，空间复杂度 : O(1)
func Merge(nums1 []int, m int, nums2 []int, n int) {
	nums1Len := len(nums1)
	nums2Len := len(nums2)
	if nums1Len == 0 || nums2Len == 0 {
		return
	}
	if nums1Len != m {
		m = nums1Len
	}
	if nums2Len != n {
		n = nums2Len
	}
	curr := m + n - 1 // 总进度指针

	for m > 0 && n > 0 {
		if nums1[m-1] > nums2[n-1] {
			nums1[curr] = nums1[m-1]
			m--
		} else {
			nums1[curr] = nums2[n-1]
			n--
		}
		curr--
	}
	for m > 0 {
		nums1[curr] = nums1[m-1]
		m--
		curr--
	}
	for n > 0 {
		nums1[curr] = nums2[n-1]
		n--
		curr--
	}
	return
}

func mergeArr(nums1 []int, m int, nums2 []int, n int) {
	i, j := m-1, n-1
	idx := m + n - 1
	for i >= 0 && j >= 0 {
		if nums1[i] > nums2[j] {
			nums1[idx] = nums1[i]
			i--
		} else {
			nums1[idx] = nums2[j]
			j--
		}
		idx--
	}
	if i >= 0 {
		return
	}
	if j >= 0 {
		copy(nums1[:idx+1], nums2[:j+1])
	}
	return
}

// 方法 2：双指针法，从前向后
// 时间复杂度: O(m + n)，空间复杂度: O(m + n), 需要一个中间数组
func Merge2(nums1 []int, m int, nums2 []int, n int) {
	if len(nums1) == 0 || len(nums2) == 0 {
		return
	}
	merged := make([]int, m+n)

	var curr, curr1, curr2 int
	for curr1 < m && curr2 < n {
		if nums1[curr1] < nums2[curr2] {
			merged[curr] = nums1[curr1]
			curr1++
		} else {
			merged[curr] = nums2[curr2]
			curr2++
		}
		curr++
	}
	for curr1 < m {
		merged[curr] = nums1[curr1]
		curr1++
		curr++
	}
	for curr2 < n {
		merged[curr] = nums2[curr2]
		curr2++
		curr++
	}
	copy(nums1, merged)
	return
}

/*
 * LeetCode 题 1103 分糖果 II
 * https://leetcode-cn.com/problems/distribute-candies-to-people/
 */

// 解法 1 暴力解法，不断循环
func DistributeCandies(candies int, num_people int) []int {
	res := make([]int, num_people)
	loop := 0
	candy := 0
	for candies != 0 {
		for i := 0; i < num_people; i++ {
			// i 从 0 开始
			// 第 i 位小朋友应该分到的糖果数量是（小朋友序号i+1）+ 总人数 * 第 n 轮（n 从 1 开始）
			candy = (i + 1) + num_people*loop
			// 判断剩下的糖果是否足够给这个小朋友
			if candies > candy {
				res[i] += candy
				candies -= candy
			} else {
				res[i] += candies
				candies = 0
			}
		}
		loop++
	}
	return res
}

// 解法 2 等差数列求和
// 参考 https://leetcode-cn.com/problems/distribute-candies-to-people/solution/fen-tang-guo-ii-by-leetcode-solution/

/*
 * 剑指 offer 面试题21. 调整数组顺序使奇数位于偶数前面
 * https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/
 * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
 * 示例：
 *     输入：nums = [1,2,3,4]
 *     输出：[1,3,2,4] （注：[3,1,2,4] 也是正确的答案之一）
 */
// 前后指针法
// 时间复杂度O(n),空间复杂度O(1)
// 扩展：判断指针是否做前移个后移的条件可以抽象成一个函数，这样就可以支持更多的情况，比如所有的正数放后面，负数放前面
func Exchange(nums []int) []int {
	low := 0
	high := len(nums) - 1
	for low < high {
		if nums[low]&1 == 1 { //  low 为奇数，满足最终条件，往后移动一个
			low++
			continue
		}
		if nums[high]&1 != 1 { // high 为偶数，满足最终条件，往前移动一个
			high--
			continue
		}
		nums[low], nums[high] = nums[high], nums[low]
		low++
		high--
	}
	return nums
}

/*
 * 寻找数组的中心索引
 * 中心索引：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和
 * https://leetcode-cn.com/explore/learn/card/array-and-string/198/introduction-to-array/770/
 *
 * 示例：
 * 输入: nums = [1, 7, 3, 6, 5, 6]
 * 输出: 3
 */
func PivotIndex(nums []int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	leftSum := 0
	for i := range nums {
		if sum-nums[i] == 2*leftSum {
			return i
		}
		leftSum += nums[i]
	}
	return -1
}

/*
 * 至少是其他数字两倍的最大数
 * 查找数组中的最大元素是否至少是数组中每个其他数字的两倍,如果是，则返回最大元素的索引，否则返回-1。
 * https://leetcode-cn.com/explore/learn/card/array-and-string/198/introduction-to-array/771/
 *
 * 示例：
 * 输入: nums = [3, 6, 1, 0]
 * 输出: 1
 */

// 解法 1：两遍循环
func DominantIndex(nums []int) int {
	max := 0
	idx := -1
	// 先找到最大值
	for i := range nums {
		if nums[i] > max {
			max = nums[i]
			idx = i
		}
	}
	for i := range nums {
		// 跳过自身
		if i == idx {
			continue
		}
		if max < nums[i]*2 {
			return -1
		}
	}
	return idx
}

// 解法 2：一遍循环
// 最关键的是要找到，数组中最大的数 m，和第二大的数 n，判断 m 与 2n 的大小关系
func DominantIndex2(nums []int) int {
	max := 0    // 数组中最大的数
	second := 0 // 数组中第二大的数
	idx := -1
	for i := range nums {
		if nums[i] <= max {
			if second < nums[i] {
				second = nums[i]
			}
			continue
		}
		second = max
		max = nums[i]
		idx = i
	}
	if max >= 2*second {
		return idx
	}
	return -1
}

/* leetcode T66. 加一(数组的加法)
 * https://leetcode-cn.com/problems/plus-one/

 * 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
 * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
 * 你可以假设除了整数 0 之外，这个整数不会以零开头
 *
 * 示例：
 * 输入: [1,2,3]
 * 输出: [1,2,4]
 */
// 加 1 问题最需要注意的就是，最后一位是否产生进位
func plusOne(digits []int) []int {
	dlen := len(digits)
	if dlen == 0 {
		return []int{1}
	}
	carry := 1
	for i := 0; i < dlen; i++ {
		num := digits[dlen-i-1] + carry
		digits[dlen-i-1] = num % 10
		carry = num / 10
		if carry == 0 {
			break
		}
	}
	if carry > 0 {
		res := make([]int, dlen+1)
		res[0] = 1
		copy(res[1:], digits)
		return res
	}
	return digits
}

/*
 * 只出现一次的数字
 *
 * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
 * https://leetcode-cn.com/explore/interview/card/top-interview-quesitons/261/before-you-start/1106/
 *
 * 示例：
 * 输入: [2,2,1]
 * 输出: 1
 */
// 将数组内的数字进行按位异或运算，相同的两个数异或值为 0，
// 一个数与 0 的异或运算结果是自身，
// 一个数与全 1 数的异或运算结果是自身按位取反
func SingleNumber(nums []int) int {
	res := 0
	for _, num := range nums {
		res ^= num
	}
	return res
}

/*
 * 只出现一次的数字 II
 *
 * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
 * https://leetcode-cn.com/problems/single-number-ii/
 *
 * 示例：
 * 输入: [2,2,3,2]
 * 输出: 3
 */
// 方法 1：位运算法
// 为了区分出现一次的数字和出现三次的数字，使用两个位掩码：seenOnce 和 seenTwice
// 仅当 seenTwice 未变时，改变 seenOnce
// 仅当 seenOnce 未变时，改变 seenTwice
func SingleNumber2(nums []int) int {
	seenOnce := 0
	seenTwice := 0
	for _, num := range nums {
		seenOnce = ^seenTwice & (seenOnce ^ num)
		seenTwice = ^seenOnce & (seenTwice ^ num)
	}
	return seenOnce
}

// 方法 2：可以用 hash 表统计次数，实现简单就不写了

// 方法 3  数学推导
// 3(a+b+c) - (3a+3b+c) = 2c
func SingleNumber22(nums []int) int {
	cnts := make(map[int]struct{}) // 做 set 过滤
	sum1 := 0
	sum2 := 0
	for _, num := range nums {
		sum1 += num
		if _, ok := cnts[num]; ok {
			continue
		}
		sum2 += num
		cnts[num] = struct{}{}
	}
	target := (3*sum2 - sum1) / 2
	return target
}

/*
 * leetcode T169. 多数元素（众数）
 *
 * https://leetcode-cn.com/problems/majority-element/
 * 给给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
 * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
 *
 * 示例：
 * 输入: [3,2,3]
 * 输出: 3
 */

// 方法 1：hash 表法
// 使用 hash 表来记录所有数字出现的次数
// 可以遍历一次 hash 表（尤其注意只有一个元素的数组），也可以遍历两次。
// 时间复杂度和空间复杂度都是 O(N)
func MajorityElement(nums []int) int {
	cnts := make(map[int]int)
	length := len(nums)
	if length == 0 {
		return 0
	}
	majorityCount := len(nums) / 2
	if majorityCount == 0 { // 单元素的情况
		return nums[0]
	}
	target := 0
	for _, num := range nums {
		cnts[num]++
		t, ok := cnts[num]
		if ok && t > majorityCount {
			target = num
			break
		}
	}
	return target
}

// 方法 2：排序法
// 把数组进行排序，数组中间那个数字肯定就是众数
// 快排的时间复杂度 O(NlogN)，可以使用内置的库函数
func MajorityElement2(nums []int) int {
	sort.Ints(nums)
	return nums[len(nums)/2]
}

// 方法 3：Boyer-Moore 投票算法
// cnt 实际表示的是众数比其他元素多出现的次数
// [7, 7, 5, 7, 5, 1 | 5, 7 | 5, 5, 7, 7 | 7, 7, 7, 7]
// 在遍历到数组中的第一个元素以及每个在 | 之后的元素时，candidate 都会因为 count 的值变为 0 而发生改变。
// 最后一次 candidate 的值从 5 变为 7，也就是这个数组中的众数
// 更多的解释可以参考 https://leetcode-cn.com/problems/majority-element/solution/duo-shu-yuan-su-by-leetcode-solution/
func MajorityElement3(nums []int) int {
	cnt := 0
	candidate := 0
	for _, num := range nums {
		if cnt == 0 {
			candidate = num
		}
		if candidate == num {
			cnt++
		} else {
			cnt--
		}
	}
	return candidate
}

// 方法 4: 随机法
// 时间复杂度：理论上最坏情况下的时间复杂度为 O(∞)，
// 因为如果我们的语气很差，这个算法会一直找不到众数，随机挑选无穷多次，所以最坏时间复杂度是没有上限的。
// 然而，运行的期望时间是线性的。
func MajorityElement4(nums []int) int {
	length := len(nums)
	if length == 0 {
		return 0
	}
	majorityCount := len(nums) / 2
	if majorityCount == 0 { // 单元素的情况
		return nums[0]
	}
	for {
		candidate := nums[rand.Intn(length)]
		if countOccurences(nums, candidate) > majorityCount {
			return candidate
		}
	}
}

func countOccurences(nums []int, num int) int {
	cnt := 0
	for _, v := range nums {
		if v != num {
			continue
		}
		cnt++
	}
	return cnt
}

/**
 * LeetCode T204. 计数质数
 * https://leetcode-cn.com/problems/count-primes/
 *
 * 统计所有小于非负整数 n 的质数的数量
 */
func CountPrimes(n int) int {
	isPrime := make([]bool, n)
	for i := 0; i < n; i++ {
		isPrime[i] = true
	}
	for i := 2; i*i < n; i++ { // 根据因子的对称性，i 只需要遍历到 sqt(n)
		if isPrime[i] {
			for j := i * i; j < n; j += i { // dito，j 从 i^2 开始尽可能避免重复标记
				isPrime[j] = false
			}
		}
	}
	cnt := 0
	for i := 2; i < n; i++ {
		if isPrime[i] {
			cnt++
		}
	}
	return cnt
}

// 方法 1：暴力法
// 最简单的方法是考虑给定 nums 数组的每个可能的子数组，找到每个子数组的元素总和，
// 并检查使用给定 k 获得的总和是否相等。
// 当总和等于 k 时，递增用于存储所需结果的 count。
// 时间复杂度：O(n^3)，会超时，空间复杂度：O(1)
func SubarraySum(nums []int, k int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
	}
	res := 0
	for start := 0; start < numsLen; start++ {
		for end := start + 1; end <= numsLen; end++ {
			sum := 0
			for i := start; i < end; i++ {
				sum += nums[i]
			}
			if sum == k {
				res++
			}
		}
	}
	return res
}

// 方法 2：暴力法 2
func SubarraySum2(nums []int, k int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
	}
	res := 0
	for start := 0; start < numsLen; start++ {
		sum := 0
		for end := start; end < numsLen; end++ {
			sum += nums[end]
			if sum == k {
				res++
			}
		}
	}
	return res
}

// 方法 3：前缀和解法 1
// 时间复杂度：O(n^2), 空间复杂度：O(n)
func SubarraySum3(nums []int, k int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
	}
	res := 0
	// 前缀和数组长度比 nums 多 1，0 位置数为 0
	// 构造前缀和
	presums := make([]int, numsLen+1)
	presums[0] = 0
	for i := 1; i <= numsLen; i++ {
		presums[i] = presums[i-1] + nums[i-1]
	}
	for start := 0; start < numsLen; start++ {
		for end := start + 1; end <= numsLen; end++ {

			// presums[end]-presums[start] 表示 nums[start] 到 nums[end-1]
			if presums[end]-presums[start] == k {
				res++
			}
		}
	}
	return res
}

// 方法 4：前缀和解法 2
// 时间复杂度：O(n)，空间复杂度 O(n)
// 该方法是方法 3 的一个改进，方法 3 中的 2 个 for 的内层循环，寻找 end 的过程，
// 实际上可以使用一个 map 来统计不同前缀和出现的频度
func SubarraySum4(nums []int, k int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
	}
	// 前缀和：该前缀和出现的次数
	preSums := map[int]int{
		0: 1,
	}
	var sum, res int
	for i := 0; i < numsLen; i++ {
		sum += nums[i] // 和的统计
		subSum := sum - k
		if _, ok := preSums[subSum]; ok {
			res += preSums[subSum]
		}
		preSums[sum]++
	}
	return res
}

/*
 * LeetCode T75. 颜色分类
 * https://leetcode-cn.com/problems/sort-colors/
 *
 * 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，
 * 使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
 * 此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
 * 注意: 不能使用代码库中的排序函数来解决这道题。
 * 示例:
 * 输入: [2,0,2,1,1,0]
 * 输出: [0,0,1,1,2,2]
 */

// 方法 1：计数排序思路
// 两次遍历
func SortColors(nums []int) {
	numsLen := len(nums)
	var red, blue, white int
	for _, color := range nums {
		switch color {
		case 0:
			red++
		case 1:
			white++
		case 2:
			blue++
		}
	}
	for i := 0; i < numsLen; i++ {
		if i < red {
			nums[i] = 0
		} else if i < red+white {
			nums[i] = 1
		} else {
			nums[i] = 2
		}
	}
}

// 方法 2：三路快排思想
// 一次遍历
func SortColors2(nums []int) {
	numsLen := len(nums)
	pRed, pBlue, curr := 0, numsLen-1, 0
	for curr < numsLen {
		if nums[curr] == 0 {
			nums[pRed], nums[curr] = nums[curr], nums[pRed]
			pRed++
			// pred 左边全是 0，所以互换后，不需要再判断 curr 的值，所以 curr++
			curr++
		} else if nums[curr] == 2 {
			nums[pBlue], nums[curr] = nums[curr], nums[pBlue]
			pBlue--
			// curr 与 blue 互换，blue 位置的数可能是 2，所以下一轮还要判断，curr 不会 ++
		} else {
			curr++
		}
	}
}

/*
 * LeetCode T350. 两个数组的交集 II
 * https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/
 *
 * 示例 1:
 * 输入: nums1 = [1,2,2,1], nums2 = [2,2]
 * 输出: [2,2]
 * 示例 2:
 * 输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
 * 输出: [4,9]
 * 说明：
 * 输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致。
 * 我们可以不考虑输出结果的顺序。
 */
// 方法 1：哈希映射
// 时间复杂度 O(m+n)，空间复杂度 O(n)
func intersect(nums1 []int, nums2 []int) []int {
	var res []int
	if len(nums1) == 0 || len(nums2) == 0 {
		return res
	}
	nums1Map := make(map[int]int)
	for _, num := range nums1 { // 可以选长度短的数组做 map
		nums1Map[num]++
	}
	for _, num := range nums2 {
		cnt := nums1Map[num]
		if cnt > 0 {
			res = append(res, num)
			nums1Map[num]--
		}
	}
	return res
}

// 方法 2：排序法
// 时间复杂度：O(nlogn+mlogm)，空间复杂度 O(1)
func intersect2(nums1 []int, nums2 []int) []int {
	var res []int
	if len(nums1) == 0 || len(nums2) == 0 {
		return res
	}
	sort.Ints(nums1)
	sort.Ints(nums2)
	len1 := len(nums1)
	len2 := len(nums2)
	for i, j := 0, 0; i < len1 && j < len2; {
		if nums1[i] < nums2[j] {
			i++
		} else if nums1[i] > nums2[j] {
			j++
		} else {
			res = append(res, nums1[i])
			i++
			j++
		}
	}
	return res
}

/*
 * LeetCode T121. 买卖股票的最佳时机
 * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
 *
 * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
 * 如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
 * 注意：你不能在买入股票前卖出股票。
 */
// 方法 1：暴力法
func maxProfit(prices []int) int {
	maxProfit := 0
	lens := len(prices)
	for i := 0; i < lens-1; i++ {
		// i 买 j 卖
		for j := i + 1; j < lens; j++ {
			profit := prices[j] - prices[i]
			if profit > maxProfit {
				maxProfit = profit
			}
		}
	}
	return maxProfit
}

// 方法 2：一次遍历法
// 找到最低历史价格
func maxProfit2(prices []int) int {
	maxProfit := 0
	minPrice := math.MaxInt32
	for _, price := range prices {
		if price < minPrice {
			minPrice = price
			continue
		}
		profit := price - minPrice // 最低价格时，为 0
		if profit > maxProfit {
			maxProfit = profit
		}
	}
	return maxProfit
}

/*
 * LeetCode T122. 买卖股票的最佳时机 II
 * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
 *
 * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
 * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
 * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）
 */
// 方法 1：贪心法
// 股票买卖策略: 单独交易日/连续上涨交易日/连续下降交易日
// 连续上涨交易日： 设此上涨交易日股票价格分别为 p1,p2..pn, 则第一天买最后一天卖收益最大, pn-p1
// 等价于每天都买卖,即 pn-p1 = (p2-p1)+(p3-p2)+...+(pn-pn-1)
// 从贪心算法思路看，就是逢低就买入，逢高就卖出。贪心算法就是说一个目光短浅的贪心的人，只会考虑下一步的得失，从不考虑长远的利益
// 时间复杂度：O(n),空间复杂度：O(1)
func maxProfit3(prices []int) int {
	maxProfit := 0
	lens := len(prices)
	for i := 0; i < lens-1; i++ {
		p := prices[i+1] - prices[i]
		if p > 0 {
			maxProfit += p
		}
	}
	return maxProfit
}

// 方法 2：峰谷法
// 时间复杂度：O(n),空间复杂度：O(1)
// 所有的波峰 - 波谷
func maxProfit4(prices []int) int {
	maxProfit := 0
	lens := len(prices)
	if lens == 0 {
		return maxProfit
	}
	peak := prices[0]
	valley := prices[0]
	for i := 0; i < lens-1; {
		for i < lens-1 && prices[i] >= prices[i+1] {
			i++
		}
		valley = prices[i]
		for i < lens-1 && prices[i] <= prices[i+1] {
			i++
		}
		peak = prices[i]
		maxProfit += peak - valley
	}
	return maxProfit
}

/*
 * LeetCode T48. 旋转图像
 * https://leetcode-cn.com/problems/rotate-image/
 *
 * 给定一个 n × n 的二维矩阵表示一个图像，将图像顺时针旋转 90 度。
 * 你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。
 */
// 时间复杂度：O(N^2)，空间复杂度：O(1)
func rotateMatrix(matrix [][]int) {
	rols := len(matrix)
	if rols == 0 {
		return
	}
	for i := 0; i < rols; i++ { // 先把矩阵转置
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
	cols := len(matrix[0])
	for i := 0; i < rols; i++ { // 然后逐行反转
		for j := 0; j < cols/2; j++ {
			matrix[i][j], matrix[i][cols-j-1] = matrix[i][cols-j-1], matrix[i][j]
		}
	}
	return
}
