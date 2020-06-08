package leetcode

/*
 * LeetCode T148. 排序链表
 * https://leetcode-cn.com/problems/sort-list/
 *
 * 在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。
 */

// 方法1：快速排序
func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	var _sort func(head, tail *ListNode)
	_sort = func(head, tail *ListNode) {
		if head == tail {
			return
		}
		pivot := listPartition(head, tail)
		_sort(head, pivot)
		_sort(pivot.Next, tail)
	}
	_sort(head, nil)
	return head
}

// 单路快排
func listPartition(head, tail *ListNode) *ListNode {
	pivot := head
	left := head
	right := head.Next
	for right != tail {
		if right.Val < pivot.Val {
			left = left.Next
			left.Val, right.Val = right.Val, left.Val
		}
		right = right.Next
	}
	pivot.Val, left.Val = left.Val, pivot.Val
	return left
}

// 方法2：（自上而下）归并排序
func sortList2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	// 1. 取中间节点
	mid := getMidNode(head)
	rHead := mid.Next
	mid.Next = nil
	// 左右两半进行排序
	left := sortList2(head)
	right := sortList2(rHead)
	// 合并左右两半有序链表
	return mergeSortedList(left, right)
}

// 假设节点序号从 1 开始，总节点数为 N
// 获得链表的中间节点为 N/2（上取整）
func getMidNode(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow := head
	fast := head
	for fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}

// 方法2：（自下而上）归并排序
func sortList3(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	// 计算链表的长度
	listLen := 0
	curr := head
	for curr != nil {
		listLen++
		curr = curr.Next
	}

	step := 1
	dummy := &ListNode{}

	// 步长不能超过链表长度
	for step < listLen {
		curr = head   // 每一轮都是从 head 开始
		tail := dummy // 已经合并的链表的尾结点
		for curr != nil {
			left := curr
			right := listCutNLen(left, step)
			curr = listCutNLen(right, step)
			// 左右 2 部分链表合并
			tail.Next = mergeSortedList(left, right)

			// 走向这一次合并后链表的尾部
			for tail.Next != nil {
				tail = tail.Next
			}
		}
		step <<= 1 // 乘法增加
	}
	return dummy.Next
}

// 在以 head 为头结点的链表中，切出长度为 len 的小链表，返回下一个节点
func listCutNLen(head *ListNode, len int) *ListNode {
	if head == nil || len <= 0 {
		return nil
	}
	var prev *ListNode
	curr := head
	for len > 0 {
		prev = curr
		curr = curr.Next
		if curr == nil {
			return nil
		}
		len--
	}
	prev.Next = nil
	return curr
}

func mergeSortedList(head1, head2 *ListNode) *ListNode {
	if head1 == nil {
		return head2
	}
	if head2 == nil {
		return head1
	}
	dummy := &ListNode{}
	curr := dummy
	curr1 := head1
	curr2 := head2
	for curr1 != nil && curr2 != nil {
		if curr1.Val < curr2.Val {
			curr.Next = curr1
			curr1 = curr1.Next
		} else {
			curr.Next = curr2
			curr2 = curr2.Next
		}
		curr = curr.Next
	}
	if curr1 != nil {
		curr.Next = curr1
	}
	if curr2 != nil {
		curr.Next = curr2
	}
	return dummy.Next
}
