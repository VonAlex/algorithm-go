package leetcode

import (
	"math/rand"
)

/*
 * leetcode 912. 数组排序
 *
 * https://leetcode-cn.com/problems/sort-an-array/
 * 给你一个整数数组 nums，请你将该数组升序排列。
 *
 * 示例：
 * 输入: nums = [5,2,3,1]
 * 输出: [1,2,3,5]
 */
// 方法 1：冒泡排序
// 每次把最大或者最小的数冒泡到数组最后
// 时间复杂度 O(n^2)
func SortArray(nums []int) []int {
	length := len(nums) - 1       // 数组最后一个坐标
	for i := 0; i < length; i++ { // 需要经过 len(nums) - 1 趟的比较
		for j := 0; j < length-i; j++ { // 这一趟需要比较的次数
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
	return nums
}

// 冒泡排序改良版 1
// 使用一个变量 swapped 标记，这次比较是否有交换过元素，
// 如果有，说明有序了，后面不用再冒泡了。
// 这样可能会减少遍历的趟数
func SortArray2(nums []int) []int {
	length := len(nums) - 1 // 数组最后一个坐标
	swapped := true
	for i := 0; i < length; i++ { // 需要经过 len(nums) - 1 趟的比较
		if !swapped {
			break
		}
		swapped = false
		for j := 0; j < length-i; j++ { // 这一趟需要比较的次数
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
				swapped = true
			}
		}
	}
	return nums
}

// 方法 2：快速排序
// 平均时间复杂度 O(nlogn)，最坏时间复杂度 O(n^2)
func SortArray3(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}

func quickSort(nums []int, lo, hi int) {
	// 优化点：在元素数量较少的时候，可以使用插入排序，减小递归的深度
	if lo >= hi {
		return
	}
	p := partition3(nums, lo, hi)
	quickSort(nums, lo, p-1)
	quickSort(nums, p+1, hi)
}

// 前后两路快排 1 (填坑)
func partition(nums []int, lo, hi int) int {
	// 同样可以做 pivot 的随机化
	pivot := nums[lo]
	l, r := lo, hi
	for l < r {
		for l < r && nums[r] > pivot {
			r--
		}
		if l < r {
			// 左边的数填右边的坑
			nums[l] = nums[r]
			l++
		}
		for l < r && nums[l] < pivot {
			l++
		}
		if l < r {
			// 右边的数填左边的坑
			nums[r] = nums[l]
			r--
		}
	}
	// pivot 填中间的坑
	nums[l] = pivot
	return l
}

// 前后两路快排 2 (交换)
// 基数 pivot 的选择很重要！
// 最坏情况下，这是一个已经排好序的数组，时间复杂度为变成 O(n^2)，所以可以随机化选择一个 pivot
func partition2(nums []int, lo, hi int) int {
	// 随机化选择基数
	randp := lo + rand.Intn(hi-lo+1)
	if randp != lo {
		nums[lo], nums[randp] = nums[randp], nums[lo]
	}
	// 或者直接选左边第一个
	// 或者选择左中右三者的中位数
	pivot := nums[lo]
	l := lo
	r := hi
	for l < r {
		// 从右边开始
		for l < r && nums[r] >= pivot {
			r--
		}
		for l < r && nums[l] <= pivot {
			l++
		}
		if l < r {
			nums[l], nums[r] = nums[r], nums[l]
		}
	}
	nums[lo], nums[l] = nums[l], nums[lo]
	return l
}

// 单路快排
func partition3(nums []int, lo, hi int) int {
	pivot := nums[lo]
	l := lo
	r := lo + 1
	for r <= hi {
		if nums[r] < pivot {
			l++
			nums[r], nums[l] = nums[l], nums[r]
		}
		r++
	}
	nums[lo], nums[l] = nums[l], nums[lo]
	return l
}

/*
 * 直接插入排序
 * 基本方法是：每一步将一个待排序的元素，按其排序码的大小，
 * 插入到前面已经排好序的一组元素的适当位置上去，直到元素全部插入为止。
 *
 * 运行时间和待排序元素的原始排序顺序密切相关，时间复杂度为 O(n^2)
 */
func InsertionSort(nums []int) {
	// i := 0
	// j := i
	length := len(nums)

	// 从第二个数开始
	// for i := 1; i < length; i++ {
	// 	j = i
	// 	temp := nums[i]
	// 	for j > 0 && nums[j-1] > temp {
	// 		nums[j] = nums[j-1]
	// 		j--
	// 	}
	// 	nums[j] = temp
	// }

	// 直接交换
	for i := 1; i < length; i++ {
		for j := i; j > 0 && (nums[j] < nums[j-1]); j-- {
			nums[j-1], nums[j] = nums[j], nums[j-1]
		}
	}
	return
}

/*
 * 希尔排序（分组插入排序）
 * 该方法的基本思想是：
 * 先将整个待排元素序列分割成若干个子序列（由相隔某个“增量”的元素组成的）分别进行直接插入排序，
 * 然后依次缩减增量再进行排序，待整个序列中的元素基本有序（增量* 足够小）时，
 * 再对全体元素进行一次直接插入排序。
 * 因为直接插入排序在元素基本有序的情况下（接近最好情况），效率是很高的
 *
 * 平均效率是 O(nlogn)
 * Shell排序比起 QuickSort，MergeSort，HeapSort慢很多。但是它相对比较简单，
 * 它适合于数据量在5000以下并且速度并不是特别重要的场合。它对于数据量较小的数列重复排序是非常好的。
 * 参考 https://blog.csdn.net/MoreWindows/article/details/6668714
 */
func ShellSort(nums []int) {
	length := len(nums)
	// 一共分 gap 组，每组 len/ gap 个元素，当 gap = 1 时，只有 1 组
	for gap := length / 2; gap > 0; gap /= 2 {
		// lg
		for i := gap; i < length; i++ {
			for j := i; j > gap-1 && (nums[j] < nums[j-gap]); j -= gap {
				nums[j-gap], nums[j] = nums[j], nums[j-gap]
			}
		}
	}
}
