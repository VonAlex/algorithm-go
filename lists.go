package leetcode

import (
	"container/list"
	"fmt"
)

// ListNode definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

/**
 * LeetCode T2 两数相加
 *
 * 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
 * 输出：7 -> 0 -> 8
 * 原因：342 + 465 = 807
 */
// 本题的关键点是链表遍历，
// 注意：1. 两个链表有长有短
//      2. 求和运算最后可能出现额外的进位
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	node := &ListNode{}
	sum := node
	carry := 0
	for l1 != nil || l2 != nil {
		if l1 != nil {
			node.Val += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			node.Val += l2.Val
			l2 = l2.Next
		}
		carry = node.Val / 10
		node.Val %= 10
		// 后面还有节点时才会创建新节点存放结果
		if l1 != nil || l2 != nil {
			node.Next = &ListNode{
				Val: carry,
			}
			node = node.Next
		}
	}
	// 考虑最后一个进位
	if carry != 0 {
		node.Next = &ListNode{
			Val: carry,
		}
	}
	return sum
}

// 使用 dummy head 简化了链表遍历的处理
func addTwoNumbers2(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	pre := dummy // 前一个节点
	carry := 0
	for l1 != nil || l2 != nil {
		x, y := 0, 0
		if l1 != nil {
			x = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			y = l2.Val
			l2 = l2.Next
		}
		sum := x + y + carry
		carry = sum / 10
		sum %= 10
		pre.Next = &ListNode{
			Val: sum,
		}
		pre = pre.Next
	}
	// 考虑最后一个进位
	if carry != 0 {
		pre.Next = &ListNode{
			Val: carry,
		}
	}
	return dummy.Next
}

/**
 * LeetCode T21. 合并两个有序链表
 * https://leetcode-cn.com/problems/merge-two-sorted-lists/
 *
 * 示例：
 * 	   输入：1->2->4, 1->3->4
 *	   输出：1->1->2->3->4->4
 */

// 方法 1：迭代法
// 时间复杂度：O(n + m), 空间复杂度：O(1)
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	dummy := &ListNode{}
	curr1 := l1
	curr2 := l2
	merged := dummy
	for curr1 != nil && curr2 != nil {
		if curr1.Val < curr2.Val {
			merged.Next = curr1
			curr1 = curr1.Next
		} else {
			merged.Next = curr2
			curr2 = curr2.Next
		}
		merged = merged.Next
	}
	if curr1 != nil {
		merged.Next = curr1
	}
	if curr2 != nil {
		merged.Next = curr2
	}
	return dummy.Next
}

// 方法 2：递归法
// 时间/空间复杂度都是O(n + m)
func mergeTwoLists2(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	if l1.Val < l2.Val {
		l1.Next = mergeTwoLists2(l1.Next, l2)
		return l1
	}
	l2.Next = mergeTwoLists2(l1, l2.Next)
	return l2
}

/*
 * 将两个递增排序的链表合并成一个递减排序的链表
 */
// 方法1： 先合并成一个递增链表，然后反正翻转
// 方法2： 头插法
func mergeTwoLists3(head1 *ListNode, head2 *ListNode) *ListNode {
	var curr, newHead, next *ListNode
	curr1 := head1
	curr2 := head2
	for curr1 != nil && curr2 != nil {
		if curr1.Val < curr2.Val { // 头插法
			curr = curr1
			curr1 = curr1.Next
		} else {
			curr = curr2
			curr2 = curr2.Next
		}
		curr.Next = newHead
		newHead = curr
	}

	if curr1 != nil {
		curr = curr1
	} else {
		curr = curr2
	}
	for curr != nil {
		next = curr.Next
		curr.Next = newHead
		newHead = curr // 更新 head
		curr = next
	}
	return newHead
}

/**
 * 剑指 offer 面试题06 从尾到头打印链表
 * 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
 * https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/
 *
 * 输入：head = [1,3,2]
 * 输出：[2,3,1]
 */

// 解法1 使用栈结构，这是最容易想到的
// 时间复杂度和空间复杂度都是 O(n)
func reversePrint(head *ListNode) []int {
	if head == nil {
		return nil
	}
	stack := list.New()
	walk := head
	for walk != nil {
		stack.PushFront(walk.Val) // 使用栈结构
		walk = walk.Next
	}
	revers := []int{}
	for e := stack.Front(); e != nil; e = e.Next() {
		revers = append(revers, e.Value.(int))
	}
	return revers
}

// 解法2 两次循环，先获得 list 的节点个数，然后倒着放置节点
// 时间复杂度 O(n)，空间复杂度 O(1)
func reversePrint2(head *ListNode) []int {
	if head == nil {
		return nil
	}
	nodeCnt := 0
	curr := head
	// 第一次遍历 list，统计链表中有多少个节点
	for curr != nil {
		nodeCnt++
		curr = curr.Next
	}
	revers := make([]int, nodeCnt)
	curr = head
	// 再次遍历 list，从后往前放 val
	for i := nodeCnt - 1; i >= 0; i-- {
		revers[i] = curr.Val
		curr = curr.Next
	}
	return revers
}

// 解法3 翻转数组
func reversePrint3(head *ListNode) []int {
	if head == nil {
		return nil
	}
	revers := []int{}
	curr := head
	// 遍历 list，将 val 顺序放入结果
	for curr != nil {
		revers = append(revers, curr.Val)
		curr = curr.Next
	}
	nodeCnt := len(revers)
	// 将结果数组进行翻转
	for i, j := 0, nodeCnt-1; i < j; {
		revers[i], revers[j] = revers[j], revers[i]
		i++
		j--
	}
	return revers
}

// 解法4 翻转链表
// 这个方法有个弊端，就是会修改原始 list，这很有可能不合题意
func reversePrint4(head *ListNode) []int {
	if head == nil {
		return nil
	}
	// 翻转链表
	var prev, next *ListNode
	curr := head
	for curr != nil {
		next = curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	revers := []int{}
	curr = prev
	for curr != nil {
		revers = append(revers, curr.Val)
		curr = curr.Next
	}
	// 可以再把 list 翻转过来
	return revers
}

// 解法5 递归法
// 当链表过长时，可能出现函数调用栈溢出的异常
func reversePrint5(head *ListNode) []int {
	if head == nil {
		return nil
	}
	revers := []int{}
	curr := head
	nextVal := reversePrintHelper(curr, &revers)
	revers = append(revers, nextVal)
	return revers
}

func reversePrintHelper(node *ListNode, res *[]int) int {
	if node.Next == nil { // 是否到了最后一个节点
		return node.Val
	}
	nextVal := reversePrintHelper(node.Next, res)
	*res = append(*res, nextVal)
	return node.Val
}

/*
 * LeetCode T206 反转链表
 * https://leetcode-cn.com/problems/reverse-linked-list/
 * 面试题24. 反转链表
 * https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/
 * 输入: 1->2->3->4->5->NULL
 * 输出: 5->4->3->2->1->NULL
 */

// 解法 1 迭代
// 时间复杂度 O(n)，空间复杂度 O(1)
func reverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	var prev, next *ListNode
	curr := head
	for curr != nil {
		// next = curr.Next
		// curr.Next = prev
		// prev = curr
		// curr = next

		next = curr.Next
		if prev == nil { // curr 是头结点
			curr.Next = nil
		} else {
			curr.Next = prev // curr 不是头结点
		}
		prev = curr
		curr = next
	}
	return prev
}

// 解法 2 递归
// 时间复杂度 O(n)，空间复杂度 O(n)
// 核心是假设递归函数已经把链表反转好了, 以 1->2->3->4->5 进行分析
// 在第一层栈中， head 是 1，需要保存新的反转后链表的 tail 节点，即 2
// 调用反转函数，2->3->4->5 反转成 2 ← 3 ← 4 ← 5， newHead 是 5
// 那么现在就要把 1 和新的 list 关联起来，这时候之前记录的 tail 节点派上了用场。
// 以此类推
func reverseList2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	next := head.Next // next 是新 list 的尾结点
	newHead := reverseList2(next)
	next.Next = head // 将传入的节点接在新 list 的后面
	head.Next = nil  // 清空指针域，成为新的尾结点
	return newHead   // 返回新头结点
}

// 解法 3 头插法
// 时间复杂度 O(n)，空间复杂度 O(1)
// 该解法的思路是不断把当前节点的下一个节点往头部插。下面举例说明，以 1->2->3->4->5 分析，
// curr = 1 时，将 2 插到头部，链表变成了 2->1->3->4->5
// 我们需要一个变量记住头部的位置，即 head，不断更新传入的 head 变量即可
// 同时需要记住下一个节点的位置，即 next
// 以此类推，直到 curr 的下一个节点是 nil 了，说明到最后一个节点了，此时的 curr 正好在 tail 节点上
func reverseList3(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	curr := head
	for curr.Next != nil {
		next := curr.Next
		curr.Next = next.Next // 将 next 节点扣出来
		next.Next = head      // next 节点转移到最前面
		head = next           // 更新 head 节点，因为刚才在head 前面插入了一个节点
	}
	// 此时 head 和 curr 分别成为了新 list 的头尾节点
	return head
}

// 递归实现反转前 N 个节点
func reverseN(head *ListNode, n int) *ListNode {
	var suc *ListNode
	var _healper func(*ListNode, int) *ListNode
	_healper = func(node *ListNode, n int) *ListNode {
		if n == 1 {
			// 记录第 n+1 个节点
			suc = node.Next
			return node
		}
		// 反转后的尾节点
		//      node  next
		//       ↓     ↓
		// 1  →  2  →  3  →  4
		next := node.Next
		// 以 node.next 为起点，需要反转 n-1 个节点
		newHead := _healper(next, n-1)

		// 连上 node
		next.Next = node

		// 连上未反转的节点
		node.Next = suc
		return newHead
	}
	return _healper(head, n)
}

/*
 * LeetCode T92. 反转链表 II
 * https://leetcode-cn.com/problems/reverse-linked-list-ii/
 *
 * 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
 * 输入: 1->2->3->4->5->NULL, m = 2, n = 4
 * 输出: 1->4->3->2->5->NULL
 *
 * 说明: 1 ≤ m ≤ n ≤ 链表长度。
 */
func reverseBetween(head *ListNode, m int, n int) *ListNode {
	if head == nil {
		return head
	}
	if m > n {
		return head
	}
	var prev *ListNode
	curr := head

	// 需要保存位置 m 的前一个结点，m = 1 时恰好遍历到要反转链表的第一个结点
	// 即，反转后的 tail 结点
	for curr != nil && m > 1 {
		prev = curr
		curr = curr.Next
		m--
		n--
	}
	if curr == nil {
		return head
	}
	lastTail := prev // 前一段的结尾
	tail := curr     // 反转部分的第一个结点，也即，反转后链表的尾部

	// n = 0 时，curr 恰好遍历到 n 的下一个节点，prev 就应该是位置 n
	for curr != nil && n > 0 {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
		n--
	}
	// curr 有可能会遍历到 nil
	if lastTail == nil {
		head = prev
	} else {
		lastTail.Next = prev
	}
	tail.Next = curr
	return head
}

/*
 * LeetCode T25. K 个一组翻转链表
 * https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
 *
 * 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
 * k 是一个正整数，它的值小于或等于链表的长度。
 * 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
 * 示例：
 * 给你这个链表：1->2->3->4->5
 * 当 k = 2 时，应当返回: 2->1->4->3->5
 * 当 k = 3 时，应当返回: 3->2->1->4->5
 *
 * 说明：
 * 你的算法只能使用常数的额外空间。你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
 */
// 方法 1：借助栈，时间复杂度 O(N)，空间复杂度 O(K)
func reverseKGroup(head *ListNode, k int) *ListNode {
	if k < 2 {
		return head
	}
	stack := list.New()
	var pre, next *ListNode
	curr := head
	newHead := head
	for curr != nil {
		next = curr.Next
		stack.PushBack(curr)
		if stack.Len() == k { // 每 k 个反转一次
			pre = reverseKGroupHelper(stack, pre, next)
			if newHead == head { // 第一次，更新 newHead
				newHead = curr
			}
		}
		curr = next
	}
	return newHead
}

func reverseKGroupHelper(stack *list.List, left, right *ListNode) *ListNode {
	curr := stack.Remove(stack.Back()).(*ListNode)
	if left != nil {
		left.Next = curr
	}
	for stack.Len() != 0 {
		next := stack.Remove(stack.Back()).(*ListNode)
		curr.Next = next
		curr = curr.Next
	}
	curr.Next = right
	return curr
}

// 方法 2：迭代法
func reverseKGroup2(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k < 2 {
		return head
	}

	var newHead, lastEnd *ListNode
	var count int

	curr := head
	for curr != nil {
		count++
		if count != k {
			curr = curr.Next
			continue
		}

		var start *ListNode // 本 group 从哪儿开始
		if lastEnd == nil {
			start = head
		} else {
			start = lastEnd.Next
		}
		pcurr := start

		var prev *ListNode
		nextStart := curr.Next // 下一 group 的第一个结点
		for pcurr != nextStart {
			next := pcurr.Next
			pcurr.Next = prev
			prev = pcurr
			pcurr = next
		}
		if lastEnd == nil {
			newHead = prev
		} else { // 非 group 1，连接翻转部分与前面的链表
			lastEnd.Next = prev
		}
		start.Next = pcurr // 连接翻转部分与后面的链表
		lastEnd = start    // 变更处理过的最后一个节点

		curr = pcurr
		count = 0
	}
	return newHead
}

// 使用 dummy 结点简化 head 的特殊情况
func reverseKGroup3(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k < 2 {
		return head
	}
	dummy := &ListNode{
		Next: head,
	}
	lastEnd, curr := dummy, head

	var count int
	for curr != nil {
		count++
		if count != k {
			curr = curr.Next
			continue
		}

		var prev *ListNode
		nextStart := curr.Next // 下一 group 的第一个结点
		pcurr := lastEnd.Next  // 本 group 从哪儿开始
		for pcurr != nextStart {
			next := pcurr.Next
			pcurr.Next = prev
			prev = pcurr
			pcurr = next
		}
		// pcurr 指向下一个 group 的第一个结点
		start := lastEnd.Next // 翻转后的最后一个结点，翻转前的第一个节点
		lastEnd.Next = prev   // 连接当前 group 与前面的链表
		start.Next = pcurr    // 连接当前 group 与后面的链表
		lastEnd = start       // 变更处理过的最后一个节点

		curr = pcurr // 需要重新定位 curr，因为原来的那组结点经过了翻转
		count = 0
	}
	return dummy.Next
}

// 使用已有的函数翻转链表
func reverseKGroup4(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k < 2 {
		return head
	}
	dummy := &ListNode{
		Next: head,
	}
	lastEnd, curr := dummy, head

	var count int
	for curr != nil {
		count++
		if count%k != 0 {
			curr = curr.Next
			continue
		}
		// count == k 时，curr 指向第 k 个节点
		nextStart := curr.Next // 保存 curr 的下一结点
		curr.Next = nil        // 断开

		start := lastEnd.Next             // 要翻转部分的第一个结点
		lastEnd.Next = reverseList(start) // 连接前面部分与新翻转的部分
		start.Next = nextStart            // tail 成为翻转部分的尾结点，重新连接链表的后面部分
		lastEnd = start                   // 更新 lastEnd

		curr = nextStart
	}
	return dummy.Next
}

// 方法 2：头插法
func reverseKGroup5(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k < 2 {
		return head
	}
	dummy := &ListNode{
		Next: head,
	}
	lastEnd, curr := dummy, head

	var count int
	for curr != nil {
		count++
		if count%k != 0 {
			curr = curr.Next
			continue
		}
		// 经过反转后，lastEnd.Next 的值就变了，所以先保存好 start 结点
		start, nextStart := lastEnd.Next, curr.Next

		pcurr := start
		for pcurr.Next != nextStart {
			next := pcurr.Next
			pcurr.Next = next.Next
			next.Next = lastEnd.Next
			lastEnd.Next = next
		}
		lastEnd = start
		curr = nextStart
	}
	return dummy.Next
}

// 方法 3：递归法
func reverseKGroup6(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k < 2 {
		return head
	}
	var count int
	curr := head
	for curr != nil && count < k {
		curr = curr.Next
		count++
	}
	// curr 定位到第 k+1 个结点，即下一 group 的首结点
	if count < k {
		return head
	}

	var prev *ListNode
	p := head
	for p != curr {
		next := p.Next
		p.Next = prev
		prev = p
		p = next
	}
	// 连接当前 group 与后面的部分
	head.Next = reverseKGroup6(curr, k)
	return prev
}

/*
 * 跟 LeetCode T25. K 个一组翻转链表题目类似，
 * 题目要求换成，如果节点总数不是 k 的整数倍，那么请头部剩余的节点保持原有顺序
 */
func reverseKGroup8(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k < 2 {
		return head
	}
	lens := getListLen(head)
	offset := lens % k

	curr := head
	var prev *ListNode
	for offset > 0 {
		prev = curr
		curr = curr.Next
		offset--
	}
	// 先把头部的余数节点放过，然后走入正常的逻辑，每 k 个一组进行翻转
	prev.Next = reverseKGroup(curr, k)
	return head
}

/*
 * LeetCode T234. 回文链表
 * https://leetcode-cn.com/problems/palindrome-linked-list/
 *
 * 请判断一个链表是否为回文链表。
 * 示例 1:
 * 输入: 1->2
 * 输出: false
 * 示例 2:
 * 输入: 1->2->2->1
 * 输出: true
 *
 */
// 方法 1 使用栈，借助其先入后出的特点
// 时/空间复杂度 O(N)
func isPalindromeList(head *ListNode) bool {
	if head == nil {
		return true
	}
	stack := list.New()
	curr := head
	for curr != nil {
		stack.PushBack(curr)
		curr = curr.Next
	}
	curr = head
	for curr != nil {
		node := stack.Remove(stack.Back())
		if node.(*ListNode).Val != curr.Val {
			return false
		}
		curr = curr.Next
	}
	return true
}

// 方法 2 反转链表
// 将链表后半部分反转，从两边遍历，对回文链表链表来说，这两半链表有着相同的遍历
// 时间复杂度 O(N)，空间复杂度 O(1)
func isPalindromeList2(head *ListNode) bool {
	if head == nil || head.Next == nil { // 讨论链表为空时返回 true 还是 false
		return true
	}
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	revHead := reverseList(slow) // 保存反转后的后半部分头结点，方便后面恢复链表
	currL := head                // 链表两头
	currR := revHead
	res := true
	for currL != nil && currR != nil {
		if currL.Val != currR.Val {
			res = false
			break
		}
		currL = currL.Next
		currR = currR.Next
	}
	reverseList(revHead) // 恢复后半段list
	return res
}

/*
 * LeetCode T143. 重排链表
 * https://leetcode-cn.com/problems/reorder-list/
 *
 * 给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
 * 将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…
 * 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
 * 示例 1:
 * 给定链表 1->2->3->4, 重新排列为 1->4->2->3.
 *
 */
func reorderList(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}
	// 1. 找到中间节点
	slow := head
	fast := head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	// 2. 反转后半部分
	curr := slow.Next
	slow.Next = nil
	var prev, next *ListNode
	for curr != nil {
		next = curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}

	// 3. 合并两部分
	second := prev
	first := head
	dummy := &ListNode{}
	curr = dummy
	for first != nil && second != nil {
		curr.Next = first
		first = first.Next
		curr = curr.Next
		curr.Next = second
		second = second.Next
		curr = curr.Next
	}
	// len(前半部分) >= len(后半部分)
	if first != nil {
		curr.Next = first
	}
	head = dummy.Next
	return
}

/*
 * LeetCode T61. 旋转链表
 * https://leetcode-cn.com/problems/rotate-list/
 * 给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。
 *
 * 示例 1:
 * 输入: 1->2->3->4->5->NULL, k = 2
 * 输出: 4->5->1->2->3->NULL
 *
 * 解释:
 * 向右旋转 1 步: 5->1->2->3->4->NULL
 * 向右旋转 2 步: 4->5->1->2->3->NULL
 */

// 方法1：截取法
// 将链表后 k 个节点组成的子链作为新链表的头部
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k <= 0 { // 单节点、空链表不需要翻转
		return head
	}
	lens := getListLen(head)
	k %= lens

	if k == 0 { // 整除，不需要翻转
		return head
	}
	fast := head
	for k > 0 { // fast 指针先走 k 步
		if fast == nil {
			return head
		}
		fast = fast.Next
		k--
	}

	slow := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}

	newHead := slow.Next // 新链表的头部在 slow 的下一个节点
	slow.Next = nil
	fast.Next = head
	return newHead
}

// 方法2：切环法
// 将单链表收尾相连成环，然后在 len - k%len - 1 位置切断
func rotateRight2(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k <= 0 { // 单节点、空链表不需要翻转
		return head
	}
	listLen := 1
	curr := head
	for curr.Next != nil {
		curr = curr.Next
		listLen++
	}
	curr.Next = head
	tailIdx := listLen - k%listLen - 1 // 新的 tail 节点在链表中的位置

	newTail := head
	for i := 1; i <= tailIdx; i++ {
		newTail = newTail.Next
	}
	newHead := newTail.Next
	newTail.Next = nil
	return newHead
}

/*
 * 打印两个有序链表的公共部分
 */
// 既然是有序链表，那么此题难度就很低了
func printCommonPart(head1 *ListNode, head2 *ListNode) {
	curr1 := head1
	curr2 := head2
	for curr1 != nil && curr2 != nil {
		if curr1.Val > curr2.Val {
			curr2 = curr2.Next
		} else if curr1.Val < curr2.Val {
			curr1 = curr1.Next
		} else {
			fmt.Println(curr1.Val)
			curr2 = curr2.Next
			curr1 = curr1.Next
		}
	}
}

/*
 * LeetCode T141 环形链表
 * 给定一个链表，判断链表中是否有环
 * https://leetcode-cn.com/problems/linked-list-cycle/
 */
// 方法 1：快慢指针法
// 假设非环部分长 N，环形部分长 K，那么时间复杂度为 O(N+K)，也就是 O(n)
func hasCycle(head *ListNode) bool {
	if head == nil {
		return false
	}
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		if slow == fast {
			return true
		}
	}
	return false
}

// 方法 2：哈希表法
// 时间复杂度和空间复杂度都是 O(n)
func hasCycle2(head *ListNode) bool {
	return false
}

/*
 * LeetCode T142. 环形链表 II
 * https://leetcode-cn.com/problems/linked-list-cycle-ii/
 * LeetCode 面试题 02.08. 环路检测
 * https://leetcode-cn.com/problems/linked-list-cycle-lcci/
 *
 * 给定一个有环链表，实现一个算法返回环路的开头节点。
 */
// 方法 1：快慢指针法
func detectCycle(head *ListNode) *ListNode {
	slow := head
	fast := head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast { // 找到相遇的位置
			break
		}
	}
	// 处理没有环的情况
	if fast == nil || fast.Next == nil {
		return nil
	}
	fast = head // fast 从头开始走
	for slow != fast {
		slow = slow.Next
		fast = fast.Next
	}
	// 相遇到交点处
	return slow
}

/*
 * LeetCode 面试题52. 两个链表的第一个公共节点
 * https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/
 * LeetCode 面试题 02.07. 链表相交
 * https://leetcode-cn.com/problems/intersection-of-two-linked-lists-lcci/
 *
 * 输入两个链表，找出它们的第一个公共节点。
 */
// 方法 1：双指针法
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	currA := headA
	currB := headB
	if currA == nil || currB == nil {
		return nil
	}
	var end int
	for currA != currB {
		currA = currA.Next
		if currA == nil {
			currA = headB
			end++
		}
		currB = currB.Next
		if currB == nil {
			currB = headA
			end++
		}
		// 当两个指针同时到达结尾时，end = 4，没有公共部分，返回 nil
		if end > 2 {
			return nil
		}
	}
	return currA
}

// 方法 2：哈希表法

/*
 * 求两个链表的第一个公共节点。
 * 两个链表，可能有环，可能无环
 */
func getListIntersection(headA, headB *ListNode) *ListNode {
	loopA := detectCycle(headA)
	loopB := detectCycle(headB)
	if loopA == nil && loopB == nil { // AB 都无环
		return getIntersectionNode(headA, headB)
	}
	if loopA != nil && loopB != nil { // AB 都有环
		if loopA == loopB { // 入环处相同
			return loopA
		}
		// 入环处不同
		curr := loopA.Next
		for curr != loopA {
			if curr == loopB { // curr 在环内转了一圈，能找到 loopB，说明 loopB 也在环上
				return curr
			}
			curr = curr.Next
		}
		return nil // loopB 不在环上，说明不相交
	}
	return nil // 一个有环，一个无环
}

/*
 * LeetCode T876 链表的中间结点
 * https://leetcode-cn.com/problems/middle-of-the-linked-list/
 *
 * 给定一个带有头结点 head 的非空单链表，返回链表的中间结点。
 * 如果有两个中间结点，则返回第二个中间结点。
 */
// 方法 1：快慢指针法
// 时间复杂度 O(N)，空间复杂度 O(1)
func middleNode(head *ListNode) *ListNode {
	slow := head
	fast := head
	// 注意奇数节点和偶数节点的终止条件是不同的
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}

// 方法 2：单指针法
// 先遍历一遍获得长度，第二遍遍历找到中间节点

/*
 * LeetCode 面试题 02.03. 删除链表中间的某个节点
 * https://leetcode-cn.com/problems/delete-middle-node-lcci/
 * LeetCode T237. 删除链表中的节点
 * https://leetcode-cn.com/problems/delete-node-in-a-linked-list/
 *
 * 实现一种算法，删除单向链表中间的某个节点（除了第一个和最后一个节点，不一定是中间节点），假定你只能访问该节点。
 * 示例：
 * 输入：单向链表a->b->c->d->e->f中的节点c
 * 结果：不返回任何数据，但该链表变为a->b->d->e->f
 */
// 实例本题提示了一种删除链表节点的方法，不要找到前一个节点，而是后一个节点覆盖要删除的节点，然后改变指针
func deleteSomeNode(node *ListNode) {
	if node == nil || node.Next == nil {
		return
	}
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

// 删除链表的中间节点
// Q1：确认偶数节点的时候删哪个（前 or 后）
// Q2: 确认只有一个节点怎么处理
func deleteMiddleNode(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	slow, fast := head, head
	if fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	if slow.Next == nil {
		head.Next = nil
	} else {
		slow.Val = slow.Next.Val
		slow.Next = slow.Next.Next
	}
	return head
}

// 合理使用 prev 辅助节点
// ↓
// 1 → 2 → 3 → 4（偶数）
//         ↑
//
//         ↓
// 1 → 2 → 3 → 4 → 5（奇数）
//                 ↑
func deleteMiddleNode2(head *ListNode) *ListNode {
	var prev *ListNode
	slow, fast := head, head

	for fast != nil && fast.Next != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	if prev == nil {
		return nil
	}
	prev.Next = slow.Next
	return head
}

/*
 * 剑指 offer 面试题 18 删除链表的节点
 * https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/
 *
 * 输入: head = [4,5,1,9], val = 5
 * 输出: [4,1,9]
 *
 * 说明：题目保证链表中节点的值互不相同
 */
func deleteNode(head *ListNode, val int) *ListNode {
	if head == nil {
		return head
	}
	// 使用 dummy 节点简化特殊情况的处理
	dummy := &ListNode{
		Next: head,
	}
	prev, curr := dummy, head
	for curr != nil {
		if curr.Val == val {
			prev.Next = curr.Next
			break
		}
		prev = curr
		curr = curr.Next
	}
	return dummy.Next
}

/*
 * LeetCode T203. 删除链表中等于给定值 val 的所有节点。
 * https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/
 *
 * 输入: 1->2->6->3->4->5->6, val = 6
 * 输出: 1->2->3->4->5
 *
 * 说明：链表中有相同的元素
 */
// 本题是上面一题的升级版
// 方法 1: 不使用 dummy 结点
func removeElements(head *ListNode, val int) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		if curr.Val == val {
			if prev == nil { // 要删除的是头部节点
				head = head.Next
			} else {
				prev.Next = curr.Next
			}
		} else {
			prev = curr
		}
		curr = curr.Next
	}
	return head
}

// 方法 2：使用 dummy 结点简化特殊情况
func removeElements2(head *ListNode, val int) *ListNode {
	dummy := &ListNode{
		Next: head,
	}
	prev, curr := dummy, head
	for curr != nil {
		if curr.Val == val {
			prev.Next = curr.Next // prev 不动
		} else {
			prev = curr
		}
		curr = curr.Next
	}
	return dummy.Next
}

/*
 * LeetCode T83. 删除排序链表中的重复元素
 * https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/
 *
 * 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
 *
 * 示例 1:
 * 输入: 1->1->2
 * 输出: 1->2
 *
 * 示例 2:
 * 输入: 1->1->2->3->3
 * 输出: 1->2->3
 */
// 解法 1
func deleteDuplicates(head *ListNode) *ListNode {
	curr := head
	for curr != nil && curr.Next != nil {
		if curr.Val == curr.Next.Val {
			curr.Next = curr.Next.Next
		} else {
			curr = curr.Next
		}
	}
	return head
}

// 解法 2
func removeDuplicateNodes(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	pre := head
	cur := head.Next
	ex := make(map[int]bool)
	ex[pre.Val] = true
	for cur != nil {
		if _, ok := ex[cur.Val]; ok {
			pre.Next = cur.Next
			cur = pre.Next
		} else {
			ex[cur.Val] = true
			pre = pre.Next
			cur = cur.Next
		}
	}
	return head
}

/*
 * LeetCode T82. 删除排序链表中的重复元素 II
 * https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/
 *
 * 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中没有重复出现的数字
 *
 * 示例 1:
 * 输入: 1->2->3->3->4->4->5
 * 输出: 1->2->5
 *
 * 示例 2:
 * 输入: 1->1->1->2->3
 * 输出: 2->3
 */
// 方法 1: 迭代法
func deleteDuplicates2(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	dummy := &ListNode{
		Next: head,
	}
	prev, curr := dummy, head
	for curr != nil && curr.Next != nil {
		// 跳过所有重复数字
		for curr.Next != nil && curr.Val == curr.Next.Val {
			curr = curr.Next
		}
		if prev.Next == curr {
			prev = curr
		} else { // 有连续相等的结点，curr 发生的移动，跳过它们
			prev.Next = curr.Next
		}
		curr = curr.Next
	}
	return dummy.Next
}

// 方法 2: 递归法
func deleteDuplicates3(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	// 重复结点
	if head.Next != nil && head.Val == head.Next.Val {
		// 跳过所有重复数字
		for head.Next != nil && head.Val == head.Next.Val {
			head = head.Next
		}
		// 处理重复数字后的下一个结点
		return deleteDuplicates3(head.Next)
	}
	// 非重复结点，保留当前结点，处理下一个
	head.Next = deleteDuplicates3(head.Next)
	return head
}

/**
 * 剑指 offer 面试题22. 链表中倒数第k个节点
 * https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/
 * 程序员面试金典 面试题 02.02. 返回倒数第 k 个节点
 * https://leetcode-cn.com/problems/kth-node-from-end-of-list-lcci/
 *
 * 输入一个链表，输出该链表中倒数第 k 个节点。
 * 为了符合大多数人的习惯，本题从 1 开始计数，即链表的尾节点是倒数第 1 个节点。
 */
// 方法 1：快慢指针法
func getKthFromEnd(head *ListNode, k int) *ListNode {
	if k <= 0 { // 参数不合理
		return head
	}
	slow, fast := head, head
	for fast != nil && k > 0 {
		fast = fast.Next
		k--
	}
	if k > 0 { // 参数不合理, k > 链表长度
		return head
	}
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
	}
	return slow
}

// 方法 2：2 次遍历法
// 第 1 遍遍历得到链表长度为 n，第 2 遍遍历找到第 n-k 个节点

/*
 * LeetCode T19 删除单链表的倒数第 K 个节点
 * 示例：
 * 给定一个链表: 1->2->3->4->5, 和 n = 2.
 * 当删除了倒数第二个节点后，链表变为 1->2->3->5.
 * 给定的 n 保证是有效的。
 *
 * 考察点是链表的删除，重要的是找到要删除节点的前一个节点
 */

// 解法 1：两遍遍历链表1
// 假设链表长度是 N，倒数第 K 个节点的前一个节点就是第 N - K 个节点
// 第 1 次遍历，每个节点 -1，到最后一个节点，该值为 K - N
// 第 2 次遍历，每个节点 +1，当值为 0 时，就找到了要找的节点
func removeLastKthNode(head *ListNode, n int) *ListNode {
	if head == nil || n <= 0 {
		return head
	}
	curr := head
	for curr.Next != nil {
		curr = curr.Next
		n--
	}
	if n == 1 {
		return head.Next
	}
	if n > 1 {
		return head
	}
	curr = head
	for n < 0 {
		curr = curr.Next
		n++
	}
	curr.Next = curr.Next.Next
	return head
}

// 解法 2：两遍遍历链表 2 (跟上面的解法类似，核心是找到第 N - K + 1 个节点)
func removeLastKthNode2(head *ListNode, n int) *ListNode {
	if head == nil || n <= 0 {
		return head
	}
	curr := head
	length := 0
	for curr != nil {
		curr = curr.Next
		length++
	}
	length -= n
	if length < 0 {
		return head
	}
	if length == 0 {
		return head.Next
	}
	curr = head
	for length > 1 {
		curr = curr.Next
		length--
	}
	curr.Next = curr.Next.Next
	return head
}

// 解法 3：快慢指针
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	if n <= 0 {
		return head
	}
	slow, fast := head, head
	for fast != nil && n > 0 {
		fast = fast.Next
		n--
	}
	if n > 0 {
		return head
	}
	if fast == nil {
		return head.Next
	}

	var prev *ListNode
	for fast != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next
	}
	prev.Next = slow.Next
	return head
}

/*
 * LeetCode T328. 奇偶链表
 * https://leetcode-cn.com/problems/odd-even-linked-list/
 *
 * 给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。
 * 请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。
 * 请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。
 */
// 分别考虑链表奇偶数个节点
//   ______           ______
// |       ↓        |       ↓
// 1 → 2 → 3        1 → 2 → 3 → 4
//     | __↑            |_______↑
func oddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	odd := head
	evenHead := head.Next
	even := evenHead
	for odd.Next != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = nil // 清理最后一个节点
	odd.Next = evenHead
	return head
}

// 分离出奇数节点和偶数节点组成的单链表
func divideOddEvenList(head *ListNode) (oddHead, evenHead *ListNode) {
	if head == nil {
		return
	}
	if head.Next == nil {
		oddHead = head
		return
	}
	oddHead = head
	odd := oddHead
	evenHead = head.Next
	even := evenHead
	for odd.Next != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = nil
	return
}

/*
 * LeetCode T138. 复制带随机指针的链表
 * https://leetcode-cn.com/problems/copy-list-with-random-pointer/
 * LeetCode 面试题35. 复杂链表的复制
 * https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/
 *
 * 给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。
 * 要求返回这个链表的深拷贝。
 */
type ComplexNode struct {
	Val    int
	Next   *ComplexNode
	Random *ComplexNode
}

// 方法 1：时间复杂度 O(N)，空间复杂度 O(N)
func copyRandomList(head *ComplexNode) *ComplexNode {
	if head == nil {
		return head
	}
	copyHead := &ComplexNode{
		Val: head.Val,
	}
	curr := head
	prev := copyHead // 要有一个 prev 节点，来 next 连接 copy 节点
	curr = curr.Next
	// 1. 复制有一个 random 为空的单链表
	for curr != nil {
		copyNode := &ComplexNode{
			Val: curr.Val,
		}
		prev.Next = copyNode
		prev = prev.Next
		curr = curr.Next
	}
	// 2. 映射 A → A'
	nodes := make(map[*ComplexNode]*ComplexNode)
	curr = head
	copyCurr := copyHead
	for curr != nil {
		nodes[curr] = copyCurr
		curr = curr.Next
		copyCurr = copyCurr.Next
	}
	// 3. 根据映射，赋值 random
	curr = head
	copyCurr = copyHead
	for curr != nil {
		if curr.Random != nil {
			copyCurr.Random = nodes[curr.Random]
		}
		curr = curr.Next
		copyCurr = copyCurr.Next
	}
	return copyHead
}

// 方法 2：奇偶分离法
// 时间复杂度 O(N)，空间复杂度O(1)
func copyRandomList2(head *ComplexNode) *ComplexNode {
	if head == nil {
		return head
	}
	curr := head
	// 1. 先拷贝当前节点，作为当前节点的下一个节点
	// A → A' → B → B‘
	for curr != nil {
		currCloned := &ComplexNode{
			Val:  curr.Val,
			Next: curr.Next,
		}
		curr.Next = currCloned
		curr = currCloned.Next
	}
	curr = head
	// 2. 将 copy 节点的 random 指针赋值
	for curr != nil {
		currCloned := curr.Next
		if curr.Random != nil {
			currCloned.Random = curr.Random.Next
		}
		curr = currCloned.Next
	}
	// 3. 分离奇偶数节点组成的链表
	odd := head
	evenHead := head.Next
	even := evenHead
	for odd.Next != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = nil
	return evenHead
}

/*
 * 头条面试题:
 * 一个链表，奇数位升序偶数位降序，让链表变成升序的。
 * 例如：1->8->2->7->3->6->4->5，变为 1->2->3->4->5->6->7->8
 */
func oddEvenSortlist(head *ListNode) *ListNode {
	// 1. 抽出奇数节点组成的升序链表 L1 和偶数节点组成的降序链表 L2
	odd, even := divideOddEvenList(head)
	// 2. 将 L2 反转成升序链表
	even = reverseList(even)
	// 3. 合并 L1 和 L2 两个有序链表
	newHead := mergeTwoLists(odd, even)
	return newHead
}
