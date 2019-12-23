package leetcode

// ListNode definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

/**
 * 题2 两数相加
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
