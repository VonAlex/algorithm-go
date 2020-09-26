package leetcode

/*
 * LeetCode T1704. 二分查找
 * https://leetcode-cn.com/problems/binary-search/
 *
 * 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target ，
 * 写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
 */
// 时间复杂度 O(logN)
func binarySearch(nums []int, target int) int {
	l, r := 0, len(nums)-1
	// 搜索区间 [0, len(nums)-1]，循环结束条件 l > r
	// 能覆盖住 nums[l] = nums[r] = target 的情况
	for l <= r {
		mid := l + (r-l)>>1
		if nums[mid] == target {
			return mid
		}
		if nums[mid] > target {
			r = mid - 1 // mid 判断过了，所以 mid -1
		} else {
			l = mid + 1
		}
	}
	return -1
}

// 参考 https://leetcode-cn.com/problems/binary-search/solution/er-fen-cha-zhao-xiang-jie-by-labuladong/
// 二分查找，寻找数组中相同元素的左边界 (不断收缩右边界)
// [1, 2, 2, 2, 3] -> 1
func binarySearchFindLeftBound(nums []int, target int) int {
	numsLen := len(nums)
	// 异常考虑
	if nums[0] > target || nums[numsLen-1] < target {
		return -1
	}
	l, r := 0, numsLen-1
	for l <= r {
		mid := l + (r-l)>>1
		if nums[mid] == target {
			r = mid - 1
		} else if nums[mid] > target {
			r = mid - 1
		} else {
			l = mid + 1
		}
	}
	return l
}

// 二分查找，寻找数组中相同元素的右边界 (不断收缩左边界)
// [1, 2, 2, 2, 3] -> 3
func binarySearchFindRightBound(nums []int, target int) int {
	numsLen := len(nums)
	// 异常考虑
	if nums[0] > target || nums[numsLen-1] < target {
		return -1
	}
	l, r := 0, numsLen-1
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
	return r
}

/*
 * LeetCode T33. 搜索旋转排序数组
 * https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
 *
 * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
 * ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
 *
 * 搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
 * 你可以假设数组中不存在重复的元素。
 * 你的算法时间复杂度必须是 O(log n) 级别。
 */
func search(nums []int, target int) int {
	l, r := 0, len(nums)-1
	for l <= r {
		mid := l + (r-l)>>1
		if nums[mid] == target {
			return mid
		}
		if nums[mid] >= nums[l] { // 左半边有序
			if nums[l] <= target && nums[mid] > target { // 包含左边界
				r = mid - 1
			} else {
				l = mid + 1
			}
		} else { // 右半边有序
			if target > nums[mid] && target <= nums[r] { // 包含右边界
				l = mid + 1
			} else {
				r = mid - 1
			}
		}
	}
	return -1
}

/*
 * LeetCode T81. 搜索旋转排序数组 II
 * https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/
 */
// 该题跟上一题类似，不过允许数组里有重复元素
func search2(nums []int, target int) bool {
	if len(nums) == 0 {
		return false
	}
	l, r := 0, len(nums)-1
	for l <= r {
		mid := l + (r-l)>>1
		if nums[mid] == target {
			return true
		}
		if nums[mid] == nums[l] {
			l++
			continue
		}
		if nums[mid] > nums[l] { // 左半边有序
			if nums[l] <= target && nums[mid] > target { // 包含左边界
				r = mid - 1
			} else {
				l = mid + 1
			}
		} else { // 右半边有序
			if target > nums[mid] && target <= nums[r] { // 包含右边界
				l = mid + 1
			} else {
				r = mid - 1
			}
		}
	}
	return false

}

/*
 * LeetCode T153. 寻找旋转排序数组中的最小值
 * https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/
 *
 * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
 * 请找出其中最小的元素。
 *
 * 你可以假设数组中不存在重复元素。
 * 示例 1:
 * 输入: [3,4,5,1,2]
 * 输出: 1
 */
// 特殊 case：已排序数组（包含单元素数组）
func findMin(nums []int) int {
	l, r := 0, len(nums)-1
	for l < r { // l < r 是为了规避已排序数组是 mid+1或者mid-1出现数组越界的情况
		mid := l + (r-l)>>1
		if nums[mid] > nums[mid+1] {
			return nums[mid+1]
		}
		if nums[mid] < nums[mid-1] {
			return nums[mid]
		}
		if nums[mid] > nums[r] {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return nums[l]
}

/*
 * LeetCode T154. 寻找旋转排序数组中的最小值 II
 * 153 题中，如果数组中有重复元素呢？
 */
// 如下图所示的旋转数组
//                 o  o  o
//              o
//           o
//  o  o  o                       o  o  o  o
//                          o  o
// 可以发现 153 题是本题的一个特殊情况
// 在最坏情况下，也就是数组中包含相同元素时(nums[mid]==nums[r])，需要逐个遍历元素，复杂度为 O(N)
// 本题题解同样适用于上一题，只是在上一题中遇到特殊情况会提前结束循环
func findMin2(nums []int) int {
	l, r := 0, len(nums)-1
	// l = r 时退出，即变化点
	for l < r {
		mid := l + (r-l)>>1
		if nums[mid] == nums[r] { // 此时无法知道应该缩减左半边还是右半边，为防止漏掉变化点， r--
			r--
			// 不能使用 mid 与 l 进行比较，无法区分 1,2,3,4,5,6 这种排序好数组的 case
		} else if nums[mid] > nums[r] { // 缩减左半边
			l = mid + 1
		} else if nums[mid] < nums[r] { // 缩减右半边，mid 可能是变化点，防止漏掉，r = mid
			r = mid
		}
	}
	return nums[l]
}

/*
 * LeetCode T69. x 的平方根
 * https://leetcode-cn.com/problems/sqrtx/
 * 实现 int sqrt(int x) 函数。
 * 计算并返回 x 的平方根，其中 x 是非负整数。
 * 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
 * 示例：
 * 输入: 4
 * 输出: 2
 * 输入: 8
 * 输出: 2
 * 说明: 8 的平方根是 2.82842...,
 * 由于返回类型是整数，小数部分将被舍去。
 */
func MySqrt(x int) int {
	l := 0
	r := x
	// 搜索区间是 [0, x]
	for l <= r {
		mid := l + (r-l)>>1
		target := mid * mid
		if target == x {
			return mid
		} else if target > x {
			r = mid - 1
		} else if target < x {
			l = mid + 1
		}
	}
	return r // 返回右边界
}

/*
 * LeetCode T287. 寻找重复数
 * https://leetcode-cn.com/problems/find-the-duplicate-number/
 * 给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。
 * 假设只有一个重复的整数，找出这个重复的数。
 * 示例 1:
 * 输入: [1,3,4,2,2]
 * 输出: 2
 *
 * 说明：
 * 不能更改原数组（假设数组是只读的）。
 * 只能使用额外的 O(1) 的空间。
 * 时间复杂度小于 O(n^2) 。
 * 数组中只有一个重复的数字，但它可能不止重复出现一次。
 */
// 方法 1：二分法
// 元素范围 [1,n]，中位数 mid，根据 mid 与数组中小于等于 mid 的元素数量，判断缩减左半边还是右半边
// 时间复杂度 O(NlogN)--> 循环的复杂度是 O(N)
func FindDuplicate(nums []int) int {
	if len(nums) == 0 {
		return -1
	}
	l := 1
	r := len(nums) - 1
	for l < r {
		mid := l + (r-l)>>1
		lessCnt := 0
		for _, i := range nums {
			if i <= mid { // 包含 mid
				lessCnt++
			}
		}
		if lessCnt > mid { // 缩减右半边
			r = mid // 有可能 mid 是那个重复的数字，所以 r = mid 而不是 mid -1
		} else { // 缩减左半边
			l = mid + 1
		}
	}
	return l
}

// 方法 2：快慢指针法
// 数组下标n和数nums[n]建立一个映射关系f(n)
// 有重复的数，那么就肯定有多个索引指向同一个数，那么问题就转换成求有环链表的环入口
func FindDuplicate2(nums []int) int {
	if len(nums) == 0 {
		return -1
	}
	slow := nums[0]
	fast := nums[nums[0]]
	for slow != fast {
		slow = nums[slow]       // 慢指针
		fast = nums[nums[fast]] // 快指针
	}
	curr1 := 0
	curr2 := slow
	for curr1 != curr2 {
		curr1 = nums[curr1]
		curr2 = nums[curr2]
	}
	return curr1
}

// 方法 3：先排序，再找重复的数，不符合约束条件
// 方法 4：借助哈希表，不符合约束条件

/*
 * LeetCode T35. 搜索插入位置
 * https://leetcode-cn.com/problems/search-insert-position/
 * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
 * 如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
 * 你可以假设数组中无重复元素。
 * 示例：
 * 输入: [1,3,5,6], 5
 * 输出: 2
 * 输入: [1,3,5,6], 7
 * 输出: 4
 * 输入: [1,3,5,6], 0
 * 输出: 0
 */
func SearchInsert(nums []int, target int) int {
	numsLen := len(nums)
	if numsLen == 0 {
		return 0
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
	// 数组中没有 target 时返回左边界
	// case1：数组中的数都小于 target，l = numsLen + 1
	// case2：数组中的数都小于 target，l = 0(起始值)
	// case3：target 在数组中某两个数之间，l= 第一个大于 target 的数，此时 right = 最后一个小于 target 的数
	return l
}
