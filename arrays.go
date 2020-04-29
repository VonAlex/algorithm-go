package leetcode

import (
	"math/rand"
	"sort"
)

/**
 * LeetCode T1 两数之和
 * 给定一个整数数组 nums 和一个目标值 target，
 * 请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。
 */
// 这里的数组应该是不含相同元素的数组
// TwoSum 暴力法
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
// 方法 1：双指针法
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
	for l < numsLen {
		nums[l] = 0
		l++
	}
	return
}

// 方法 2：快慢指针法
// 如果当前元素是非 0 的，那么它的正确位置最多可以是当前位置或者更早的位置。
// 如果是后者，则当前位置最终将被非 0 或 0 占据，该非 0 或 0 位于大于 “cur” 索引的索引处。
// 我们马上用 0 填充当前位置，这样不像以前的解决方案，我们不需要在下一个迭代中回到这里。
// 慢指针（l）之前的所有元素都是非零的
// 当前指针和慢速指针之间的所有元素都是零。
// 时间复杂度 O(n)，需要做数组写入的次数为数组中非 0 元素的个数
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
 * 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
 * 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
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

/**
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
	l := 0
	r := 1 // 第一个元素不能修改
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

// 解法1 双指针
// 充分利用 A 和 B 已经排好序的特点
// 时间复杂度和空间复杂度都是 O(m+n)
func Merge(A []int, m int, B []int, n int) {
	C := make([]int, m+n) // 准备一个新数组，足以容纳 A + B
	var iA, iB, iC int
	for ; iC < m+n; iC++ {
		if iA == m || iB == n { // 遍历完 A 或者 B 为止
			break
		}
		if A[iA] < B[iB] {
			C[iC] = A[iA]
			iA++
		} else {
			C[iC] = B[iB]
			iB++
		}
	}
	// 遍历剩下的 A
	for ; iA < m; iA++ {
		C[iC] = A[iA]
		iC++
	}
	// 遍历剩下的 B
	for ; iB < n; iB++ {
		C[iC] = B[iB]
		iC++
	}
	copy(A, C)
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

/*
 * 加一(数组的加法)
 * 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
 * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
 * 你可以假设除了整数 0 之外，这个整数不会以零开头
 * https://leetcode-cn.com/explore/learn/card/array-and-string/198/introduction-to-array/772/
 *
 * 示例：
 * 输入: [1,2,3]
 * 输出: [1,2,4]
 */
func PlusOne(digits []int) []int {
	length := len(digits)
	if length == 0 {
		return []int{}
	}
	carry := 1 // 末尾 + 1
	lastIdx := length - 1
	for i := lastIdx; i >= 0; i-- {
		num := digits[i] + carry
		if num < 10 { // 不再产生进位了，就可以返回了
			digits[i] = num
			return digits
		}
		if num >= 10 {
			carry = num / 10
			digits[i] = num % 10 // 在原数组改，或者是新申请一个结果数组，go 是参数传值，在 digits 数组修改不会影响调用者
		}
	}
	// 最高位有进位的时候，结果肯定要多一位，原来的 digits 数组已经无法容纳了，所以需要再申请一个 +1 长度的数组
	if carry > 0 {
		res := make([]int, length+1)
		res[0] = carry
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
 * leetcode 169. 多数元素（众数）
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
