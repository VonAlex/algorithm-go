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
 * LeetCode 题2 两数相加
 *
 * 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
 * 输出：7 -> 0 -> 8
 * 原因：342 + 465 = 807
 */
// 本题的关键点是链表遍历，
// 注意：1. 两个链表有长有短
//     2. 求和运算最后可能出现额外的进位

// AddTwoNumbers ..
func AddTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
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

// AddTwoNumbers2 使用了 dummy head 简化了链表遍历的处理
func AddTwoNumbers2(l1 *ListNode, l2 *ListNode) *ListNode {
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
 * 剑指 offer 面试题06 从尾到头打印链表
 * 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
 * https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/
 *
 * 输入：head = [1,3,2]
 * 输出：[2,3,1]
 */

// 解法1 使用栈结构，这是最容易想到的
// 时间复杂度和空间复杂度都是 O(n)
func ReversePrint(head *ListNode) []int {
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
func ReversePrint2(head *ListNode) []int {
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
func ReversePrint3(head *ListNode) []int {
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
func ReversePrint4(head *ListNode) []int {
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
func ReversePrint5(head *ListNode) []int {
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

/**
 * LeetCode 题 206 反转链表
 * https://leetcode-cn.com/problems/reverse-linked-list/
 *
 * 输入: 1->2->3->4->5->NULL
 * 输出: 5->4->3->2->1->NULL
 */

// 解法 1 迭代
// 时间复杂度 O(n)，空间复杂度 O(1)
func ReverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	var prev, next *ListNode
	curr := head
	for curr != nil {
		next = curr.Next
		curr.Next = prev
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
func ReverseList2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	next := head.Next // next 是新 list 的尾结点
	newHead := ReverseList2(next)
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
func ReverseList3(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	var next *ListNode
	curr := head
	for curr.Next != nil {
		next = curr.Next
		curr.Next = next.Next // 将 next 节点扣出来
		next.Next = head      // next 节点转移到最前面
		head = next           // 更新 head 节点，因为刚才在head 前面插入了一个节点
	}
	// 此时 head 和 curr 分别成为了新 list 的头尾节点
	return head
}

/***************************** 辅助函数 *********************************/

// ListPrint 正向打印 list
func ListPrint(node *ListNode) {
	if node == nil {
		return
	}
	for node != nil {
		fmt.Print(node.Val)
		if node.Next != nil {
			fmt.Print("->")
		}
		node = node.Next
	}
	fmt.Println()
}
