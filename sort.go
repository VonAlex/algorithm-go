package leetcode

import (
	"container/heap"
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

// 参考文档 https://goa.lenggirl.com/algorithm/sort.html

// 不稳定算法记忆口诀 “快些选队” 快：快速排序 些：希尔排序 选：选择排序 队：堆排序
// 为什么区分稳定和不稳定？ https://www.cxyxiaowu.com/2573.html

// 方法 1：冒泡排序（交换排序 // 稳定）
// 每次把最大或者最小的数冒泡到数组最后
// 时间复杂度 O(n^2)
func BubbleSort(nums []int) []int {
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
func BubbleSort2(nums []int) []int {
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

// 方法 2：快速排序(交换排序)
// 平均时间复杂度 O(nlogn)，最坏时间复杂度 O(n^2)
func QuickSort(nums []int) []int {
	quickSortHelper(nums, 0, len(nums)-1)
	return nums
}

func quickSortHelper(nums []int, lo, hi int) {
	// 优化点：在元素数量较少的时候，可以使用插入排序，减小递归的深度
	if lo >= hi {
		return
	}
	p := partition3(nums, lo, hi)
	quickSortHelper(nums, lo, p-1)
	quickSortHelper(nums, p+1, hi)
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
	r := lo + 1 // l 的下一个
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

// 三路快排
func QuickSort3Way(nums []int, lo, hi int) {
	if lo >= hi {
		return
	}
	l, r := partition5(nums, lo, hi)
	QuickSort3Way(nums, lo, l-1)
	QuickSort3Way(nums, r, hi)
}

func partition5(nums []int, lo, hi int) (l, r int) {
	pivot := nums[lo]
	l = lo         // nums[lo+1...l] < pivot
	r = hi + 1     // nums[r...hi] > pivot
	curr := lo + 1 // nums[l+1...curr) == pivot
	for curr < r {
		if nums[curr] == pivot {
			curr++
		} else if nums[curr] > pivot {
			nums[curr], nums[r-1] = nums[r-1], nums[curr]
			r--
		} else {
			nums[l+1], nums[curr] = nums[curr], nums[l+1]
			curr++
			l++
		}
	}
	nums[lo], nums[l] = nums[l], nums[lo]
	return
}

/*
 * 直接插入排序 (插入排序 // 稳定)
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

/*
 * 归并排序（稳定）
 * 归并排序是一种分治策略的排序算法，通过递归地先使每个子序列有序，再将两个有序的序列进行合并成一个有序的序列。
 * 归并排序是唯一一个有稳定性保证的高级排序算法，某些时候，为了寻求大规模数据下排序前后，相同元素位置不变，可以使用归并排序。
 */
// https://goa.lenggirl.com/algorithm/sort/merge_sort.html
// 优化点 1：数组元素个数较少可以使用插入排序
// 优化点 2: 合并有序数组时需要借助额外的空间，可以使用手摇算法进行原地合并

// 自顶向下的归并排序递归实现
// 每次都是一分为二，特别均匀，所以最差和最坏时间复杂度都一样,时间复杂度为：O(nlogn)
// 递归栈的空间复杂度为：O(logn)
func MergeSort(nums []int) {
	start := 0
	end := len(nums) // end 指的是有效范围的下一个位置，即排序 [start, end) 范围内的数据
	mergeSortHelper(nums, start, end)
	return
}

func mergeSortHelper(nums []int, start, end int) {
	if end-start <= 1 { // 只有一个元素，不需要再分了
		return
	}
	mid := start + (end-start)>>1
	mergeSortHelper(nums, start, mid)
	mergeSortHelper(nums, mid, end)
	merge(nums, start, mid, end)
	return
}

// 归并操作：合并两个有序数组
func merge(nums []int, l, mid, r int) {
	lLen := mid - l
	rLen := r - mid
	mergedLen := r - l
	// 优化点1：[1, 3] 和 [4, 6] 这样的不需要进行“并”操作
	if nums[mid-1] < nums[mid] {
		return
	}
	merged := make([]int, 0, mergedLen)
	lp := 0 // 左边指针
	rp := 0 // 右边指针
	for lp < lLen && rp < rLen {
		lval := nums[l+lp]
		rval := nums[mid+rp]
		if lval < rval {
			merged = append(merged, lval)
			lp++
		} else {
			merged = append(merged, rval)
			rp++
		}
	}
	merged = append(merged, nums[l+lp:mid]...)
	merged = append(merged, nums[mid+rp:r]...)
	for i := 0; i < mergedLen; i++ { // 拷贝到原数组
		nums[l+i] = merged[i]
	}
	return
}

// 自下向上的归并排序
// 时间复杂度为：O(nlogn)，没有递归，空间复杂度为：O(1)
func MergeSort2(nums []int) {
	start := 0
	end := len(nums) // end 指的是有效范围的下一个位置，即排序 [start, end) 范围内的数据
	mergeSortHelper2(nums, start, end)
}

// 左半部分 [l. mid)，右半部分 [mid, r)
func mergeSortHelper2(nums []int, start, end int) {
	step := 1 // 起始步长为 step
	for end-start > step {
		for i := start; i < end; i += step << 1 {
			l := i
			mid := l + step  //
			r := l + step<<1 // l + 2 * step
			if mid >= end {  // 没有右半部分
				break
			}
			if r > end { // 右半部分长度不够 step
				r = end
			}
			merge(nums, l, mid, r)
		}
		step <<= 1 // 步长翻倍
	}
	return
}

/*
 * 直接选择排序（稳定）
 * 第一趟从n个元素的数据序列中选出关键字最小/大的元素并放在最前/后位置，
 * 下一趟从n-1个元素中选出最小/大的元素并放在最前/后位置。以此类推，经过n-1趟完成排序
 * 时间复杂度 O(n^2)
 */
func SelectSort(nums []int) {
	length := len(nums) - 1
	for i := 0; i < length; i++ { // 需要 length - 1 趟比较
		for j := i + 1; j <= length; j++ {
			if nums[j] < nums[i] {
				nums[i], nums[j] = nums[j], nums[i]
			}
		}
	}
	return
}

// 优化 1：不用每次都 swap，而是一趟选出最小的，与开头元素交换
func SelectSort2(nums []int) {
	length := len(nums) - 1
	for i := 0; i < length; i++ { // 需要 length - 1 趟比较
		minIdx := i // 每轮都找到最小的元素
		for j := i + 1; j <= length; j++ {
			if nums[j] < nums[minIdx] {
				minIdx = j
			}
		}
		if minIdx != i { // 交换
			nums[i], nums[minIdx] = nums[minIdx], nums[i]
		}
	}
	return
}

// 优化 2：一趟选出最大与最小，从两端进行交换
// 时间复杂度减为原来的一半
// 优化后的选择排序还是很慢，它很好理解，但是还是不建议在工程上使用
func SelectSort3(nums []int) {
	length := len(nums) - 1
	for i := 0; i < length/2; i++ {
		minIdx := i
		maxIdx := i
		for j := i + 1; j <= length-i; j++ {
			if nums[j] < nums[minIdx] {
				minIdx = j
			}
			if nums[j] > nums[maxIdx] {
				maxIdx = j
			}
		}
		// [i+1, length-i] 范围内最大值是开头的元素，最小值不是最尾的元素
		if maxIdx == i && minIdx != length-i {
			// 为了防止把开头元素换掉，先换最大值，再换最小值
			nums[length-i], nums[maxIdx] = nums[maxIdx], nums[length-i]
			nums[i], nums[minIdx] = nums[minIdx], nums[i]
			// 恰好是开头和结尾的元素，交互它们即可
		} else if maxIdx == i && minIdx == length-i {
			nums[maxIdx], nums[minIdx] = nums[minIdx], nums[maxIdx]
		} else {
			/// 中间元素的情况
			// minIdx 有可能 = i，先交换 maxIdx 就会出错
			nums[i], nums[minIdx] = nums[minIdx], nums[i]
			nums[length-i], nums[maxIdx] = nums[maxIdx], nums[length-i]
		}
	}
	return
}

/*
 * 堆排序（不稳定）
 * 堆实际上是一棵完全二叉树。
 * 堆满足两个性质:
 * 1. 堆的每一个父节点都大于（或小于）其子节点；
 * 2. 堆的每个左子树和右子树也是一个堆。
 *
 * 堆排序的步骤分为三步:
 * 1. 建堆（升序建大堆，降序建小堆）
 * 2. 交换数据
 * 3. 向下调整
 *
 * 假设二叉堆总共有n个元素，下沉调整的最坏时间复杂度等于二叉堆的高度，也就是O(logn)。--> 完全二叉树的高度是 log(n)
 * 把无序数组构建成二叉堆，需要进行n/2次循环。每次循环调用一次下沉调整方法，计算规模是n/2*logn，时间复杂度O(nlogn)。
 * 然后，在 2/3 阶段，需要循环 n-1 次，每次都要调用一次下沉调整方法，计算规模是(n-1)*logn，时间复杂度O(nlogn)
 * 因此，整体的时间复杂度为 Ο(nlogn)
 */

func HeapSort(nums []int) {
	length := len(nums)
	buildMaxHeap(nums, length)
	// 2/3 循环删除堆顶元素，移到集合尾部，调节堆产生新的堆顶
	for i := length - 1; i > 0; i-- {
		nums[0], nums[i] = nums[i], nums[0]
		length--
		heapify3(nums, 0, length)
	}
}

// 1 建堆
func buildMaxHeap(nums []int, length int) {
	// 二叉堆第一个非叶子节点
	for i := length/2 - 1; i >= 0; i-- {
		heapify3(nums, i, length)
	}
	return
}

// 递归实现堆化
func heapify(nums []int, i, length int) {
	left := 2*i + 1
	right := 2*i + 2
	largest := i
	// 选出 i,left,right 这三个位置上最大的数
	if left < length && nums[left] > nums[largest] {
		largest = left
	}
	if right < length && nums[right] > nums[largest] {
		largest = right
	}
	if largest == i {
		return
	}
	nums[largest], nums[i] = nums[i], nums[largest]
	// largest 做了交换，需要重新堆化
	heapify(nums, largest, length)
	return
}

// 迭代实现堆化
func heapify2(nums []int, i, length int) {
	for i < length {
		largest := i
		left := 2*i + 1
		right := left + 1
		if left < length && nums[left] > nums[largest] {
			largest = left
		}
		if right < length && nums[right] > nums[largest] {
			largest = right
		}
		if largest == i {
			break
		}
		nums[largest], nums[i] = nums[i], nums[largest]
		i = largest // 堆化 swap 过的节点
	}
	return
}

// 参考 go heap 源码
func heapify3(nums []int, i0, n int) bool {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 {
			break
		}
		j := j1
		if j2 := j1 + 1; j2 < n && nums[j2] > nums[j1] {
			j = j2
		}
		if nums[i] > nums[j] {
			break
		}
		nums[i], nums[j] = nums[j], nums[i]
		i = j
	}
	return i > i0
}

/* 参考 https://www.jianshu.com/p/ff1797625d66
 * 桶排序
 * 1.创建一数组，里面是各个初始为空的“桶/bucket”
 * 2.分散：遍历原始数组，将每个对象放入对应的桶中
 * 3.对每个非空桶进行排序
 * 4.收集：按顺序访问桶，并将所有元素放回原始数组中
 *
 * 如果一个数一个桶，那么对于跨度比较大的数来说，空间消耗就很大了。
 * 这里桶的划分是一个问题。
 *
 * 桶排序的适用场景：数据分布相对比较均匀或者数据跨度范围并不是很大时
 */

/*
 * 基数排序
 * 对于一组数据，我们可以按照每一位对它们进行排序
 * 先按最后一位从小到大排序,
 * 再按中间一位从小到大排序，直到最前面一位
 */

/*
 * 计数排序（改进版 2 是稳定的）
 * 参考 https://www.cxyxiaowu.com/5437.html
 * 以下两种情况不适用于 1）数组最大值和最小值差值过大 2）数组元素不是正整数
 * 假设辅助数组长度为 k，输入数组长度为 n，时间复杂度为 O(k+n)，在特定场景下比快排算法要快
 * 空间复杂度 O(n+k)，（空间换时间）
 */
// 朴素版
func CountSort(nums []int) {
	if len(nums) == 0 {
		return
	}
	max := nums[0]
	for _, num := range nums { // 1.找到待排序数组中最大的数
		if num > max {
			max = num
		}
	}
	// 创建一个长度为 max+1 的数组，使用数组下标统计出现数的出现次数
	counts := make([]int, max+1)
	for _, num := range nums {
		counts[num]++
	}
	idx := 0
	// 将统计数组counts中不为 0 的下标按照个数输出，即得到原数组的排序
	for i, cnt := range counts {
		for j := 0; j < cnt; j++ {
			nums[idx] = i
			idx++
		}
	}
	return
}

// 改进版 1
// 朴素版中以数组最大值+1做统计数组长度，有可能会浪费很多空间，
// 比如[95,91,99]，需要创建一个 100 大小的数组，浪费了前面 90+的空间
// 所以改进以 max - min + 1 做为数组的长度，统计数组中存储 offset，减少空间浪费
func CountSort2(nums []int) {
	if len(nums) == 0 {
		return
	}
	max := nums[0]
	min := nums[0]
	for _, num := range nums {
		if num > max {
			max = num
		}
		if num < min {
			min = num
		}
	}
	// 统计数组长度为 max - min +1，存储 offset
	counts := make([]int, max-min+1)
	for _, num := range nums {
		counts[num-min]++
	}
	idx := 0
	// 将统计数组counts中不为 0 的下标按照个数输出，即得到原数组的排序
	for i, cnt := range counts {
		for j := 0; j < cnt; j++ {
			nums[idx] = i + min
			idx++
		}
	}
	return
}

// 改进版 2
// 改进版 1 不稳定的排序
// 改进版 2 为稳定排序
func CountSort3(nums []int) {
	if len(nums) == 0 {
		return
	}
	max := nums[0]
	min := nums[0]
	for _, num := range nums {
		if num > max {
			max = num
		}
		if num < min {
			min = num
		}
	}
	countsLen := max - min + 1
	// 统计数组长度为 max - min +1
	counts := make([]int, countsLen)
	for _, num := range nums {
		counts[num-min]++
	}
	// 累加和，counts 数组的值就是 num 在排序数组中的位置
	for i := 1; i < countsLen; i++ {
		counts[i] += counts[i-1]
	}
	numsLen := len(nums)
	sorts := make([]int, numsLen)
	// 为了保序，从 nums 数组后面开始遍历
	for i := numsLen - 1; i >= 0; i-- {
		idx := nums[i] - min
		sorts[counts[idx]-1] = nums[i]
		counts[idx]--
	}
	copy(nums, sorts)
	return
}

/********************************************** 以上为各排序算法实现 ***********************************************/

/*
 * LeetCode T215. 数组中的第K个最大元素
 * https://leetcode-cn.com/problems/kth-largest-element-in-an-array/
 *
 * 在未排序的数组中找到第 k 个最大的元素。
 * 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
 *
 * 示例：
 * 输入: [3,2,1,5,6,4] 和 k = 2
 * 输出: 5
 */
// 方法1： 冒泡法
func findKthLargest(nums []int, k int) int {
	maxIdx := len(nums) - 1
	for i := 0; i < maxIdx; i++ { // 趟数从 0 开始
		for j := 0; j < maxIdx-i; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
		// 第 k 大的元素，是倒数第 K 个
		if i+1 == k {
			return nums[maxIdx-i]
		}
	}
	return nums[0] // 只有一个元素的情况
}

// 方法 2：快速选择法
// 本方法大致上与快速排序相同。平均时间复杂度 O(N)
// 简便起见，注意到第 k 个最大元素也就是第 N - k (数组位置)个最小元素，因此可以用第 k 小算法来解决本问题。
// 也可以将数组从大到小排列， 第 K 大在数组中的位置就是 K-1
//
func findKthLargest2(nums []int, k int) int {
	lens := len(nums)
	ksmallIdx := lens - k

	// 单路快排
	var _quick func(int, int) int
	_quick = func(lo, hi int) int {
		random := lo + rand.Intn(hi-lo+1)
		if lo != random {
			nums[lo], nums[random] = nums[random], nums[lo]
		}
		pivot := nums[lo]
		left := lo
		right := left + 1
		for right <= hi {
			if nums[right] < pivot {
				left++
				nums[left], nums[right] = nums[right], nums[left]
			}
			right++
		}
		nums[lo], nums[left] = nums[left], nums[lo]

		// 根据 l 与 ksmall 的大小决定，要继续处理左半边还是右半边
		if left == ksmallIdx {
			return nums[left]
		}
		if left < ksmallIdx {
			return _quick(left+1, hi)
		}
		return _quick(lo, left-1)
	}
	return _quick(0, lens-1)
}

type intheap []int

// 实现 sort 接口的三个方法
func (p intheap) Less(i, j int) bool {
	return p[i] < p[j]
}

func (p intheap) Len() int {
	return len(p)
}

func (p intheap) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

// 实现 heap 接口的额外方法
func (p *intheap) Pop() interface{} {
	n := len(*p)
	x := (*p)[n-1]
	*p = (*p)[:n-1]
	return x
}

func (p *intheap) Push(x interface{}) {
	*p = append(*p, x.(int))
}

// 方法 3：最小堆法
// 保持堆内的元素个数为 k，最后取出元素就是要求的目标
// 时间复杂度 : O(Nlogk)。空间复杂度 : O(k)，用于存储堆元素。
func FindKthLargest3(nums []int, k int) int {
	hp := &intheap{}
	heap.Init(hp)
	for _, num := range nums {
		heap.Push(hp, num)
		if len(*hp) > k {
			heap.Pop(hp)
		}
	}
	return heap.Pop(hp).(int)
}
