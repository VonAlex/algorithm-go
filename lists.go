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
 * LeetCode T21. 合并两个有序链表
 * https://leetcode-cn.com/problems/merge-two-sorted-lists/
 *
 * 示例：
 * 	   输入：1->2->4, 1->3->4
 *	   输出：1->1->2->3->4->4
 */

// 方法 1：迭代法
// 时间复杂度：O(n + m), 空间复杂度：O(1)
func MergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
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
func MergeTwoLists2(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	if l1.Val < l2.Val {
		l1.Next = MergeTwoLists2(l1.Next, l2)
		return l1
	}
	l2.Next = MergeTwoLists2(l1, l2.Next)
	return l2
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
func ReverseList(head *ListNode) *ListNode {
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

/**
 * LeetCode T92. 反转链表 II
 * https://leetcode-cn.com/problems/reverse-linked-list-ii/
 *
 * 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
 * 输入: 1->2->3->4->5->NULL, m = 2, n = 4
 * 输出: 1->4->3->2->5->NULL
 *
 * 说明: 1 ≤ m ≤ n ≤ 链表长度。
 */
func ReverseBetween(head *ListNode, m int, n int) *ListNode {
	if head == nil {
		return head
	}
	if m >= n { // 不合理的情况，m 应该小于 n
		return head
	}

	var prev, next *ListNode
	curr := head
	for m > 1 { // 找到反转部分的前一个节点
		prev = curr
		if curr == nil { // m 大于链表长度的情况
			return head
		}
		curr = curr.Next
		m--
		n--
	}
	con := prev                // 反转部分的前一个节点
	tail := curr               // 反转部分的第一个节点，反转后成为该部分最后一个节点
	for n > 0 && curr != nil { // 兼容 n 大于链表长度的情况
		next = curr.Next
		curr.Next = prev
		prev = curr
		curr = next
		n--
	}
	if con != nil {
		con.Next = prev
	} else {
		head = prev // 从头开始反转，即 m = 1 的情况
	}
	tail.Next = curr
	return head
}

/**
 * 剑指 offer 面试题 18 删除链表的节点
 * https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/
 *
 * 输入: head = [4,5,1,9], val = 5
 * 输出: [4,1,9]
 *
 * 说明：题目保证链表中节点的值互不相同
 */
func DeleteNode(head *ListNode, val int) *ListNode {
	if head == nil {
		return head
	}
	// 使用 dummy 节点简化特殊情况的处理
	dummy := &ListNode{
		Next: head,
	}
	prev := dummy
	curr := head
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

// 不使用 dummy 结点
func DeleteNode2(head *ListNode, val int) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		if prev == nil { // curr 是头结点
			if head.Val == val {
				return head.Next
			}
		} else { // curr 不是头结点
			if curr.Val == val {
				prev.Next = curr.Next
			}
		}
		prev = curr
		curr = curr.Next
	}
	return head
}

/**
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
func RemoveElements(head *ListNode, val int) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		if prev == nil { // curr 是头结点
			if head.Val == val {
				head = head.Next
				if head == nil {
					return nil
				}
				continue
			}
		} else { // curr 不是头结点
			if curr.Val == val {
				prev.Next = curr.Next
				curr = curr.Next // 跳过 val，prev 不变，curr 往下一个
				continue
			}
		}
		prev = curr
		curr = curr.Next
	}
	return head
}

// 方法 2：使用 dummy 结点简化特殊情况
func RemoveElements2(head *ListNode, val int) *ListNode {
	dummy := &ListNode{
		Next: head,
	}
	prev := dummy
	curr := head
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
 * 打印两个有序链表的公共部分
 */
// 既然是有序链表，那么此题难度就很低了
func PrintCommonPart(head1 *ListNode, head2 *ListNode) {
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
func RemoveLastKthNode(head *ListNode, n int) *ListNode {
	if head == nil || n < 0 {
		return head
	}
	dummy := &ListNode{
		Next: head,
	}
	curr := dummy
	for curr.Next != nil {
		n--
		curr = curr.Next
	}
	if n > 0 {
		return head
	}
	if n == 0 {
		return head.Next
	}
	curr = dummy
	for n != 0 {
		n++
		curr = curr.Next
	}
	curr.Next = curr.Next.Next
	return head
}

// 解法 2：两遍遍历链表 2 (跟上面的解法类似，核心是找到第 N - K + 1 个节点)
func RemoveLastKthNode2(head *ListNode, n int) *ListNode {
	if head == nil || n < 0 {
		return head
	}
	curr := head
	length := 0
	for curr != nil {
		length++
		curr = curr.Next
	}
	// 链表长度减去 K，可能有三种情况，>0,=0,<0
	// <0 和 = 0 为异常情况，使用 dummy 节点来归一化处理
	length -= n
	dummy := &ListNode{
		Next: head,
	}
	curr = dummy
	// 找到从链表开头数起的第 (N - K) 个结点（是前置节点）
	for length > 0 {
		length--
		curr = curr.Next
	}
	curr.Next = curr.Next.Next
	return dummy.Next
}

// 解法 3 一次遍历法
// 使用快慢指针法
func RemoveLastKthNode3(head *ListNode, n int) *ListNode {
	if head == nil || n < 0 {
		return head
	}
	dummy := &ListNode{
		Next: head,
	}
	// dummy 节点解决了 n = 链表长度这种情况，进行了归一化
	fast := dummy
	slow := dummy
	for n >= 0 { // fast 先走 n+1 步
		if fast == nil {
			return head
		}
		n--
		fast = fast.Next
	}
	for fast != nil {
		fast = fast.Next
		slow = slow.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

// 解法 4：快慢指针 - 去掉 dummy 节点
func RemoveLastKthNode4(head *ListNode, n int) *ListNode {
	if head == nil || n < 0 {
		return head
	}
	fast := head
	for n > 0 { // fast 先走 n 步
		if fast == nil {
			return head
		}
		n--
		fast = fast.Next
	}
	var prev *ListNode // prev 节点辅助
	slow := head
	for fast != nil {
		prev = slow
		fast = fast.Next
		slow = slow.Next
	}
	// 遍历结束，slow 就是倒数第 k 个节点
	if prev == nil {
		head = slow.Next
	} else {
		prev.Next = slow.Next
	}
	return head
}

/**
 * 剑指 offer 面试题22. 链表中倒数第k个节点
 * https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/
 *
 * 输入一个链表，输出该链表中倒数第k个节点。
 * 为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。
 */
// 方法 1：快慢指针法
func GetKthFromEnd(head *ListNode, k int) *ListNode {
	if head == nil || k < 0 { // 参数不合理
		return head
	}
	fast := head
	for k > 0 { // fast 先走 k 步骤
		if fast == nil { // k 大于链表长度
			return head
		}
		fast = fast.Next
		k--
	}
	slow := head
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
	}
	return slow
}

// 方法 2：2 次遍历法
// 第 1 遍遍历得到链表长度为 n，第 2 遍遍历找到第 n-k 个节点

/*
 * 程序员面试金典 面试题 02.02. 返回倒数第 k 个节点
 * https://leetcode-cn.com/problems/kth-node-from-end-of-list-lcci/
 *
 * 实现一种算法，找出单向链表中倒数第 k 个节点。返回该节点的值。
 * 输入： 1->2->3->4->5 和 k = 2
 * 输出： 4
 * 说明：给定的 k 保证是有效的。
 */
// 方法 1：快慢指针法
// 实际上，要有异常情况的考虑，head 为空是返回什么值？k 为负数返回什么值？k 超过链表长度返回什么值？
func KthToLast(head *ListNode, k int) int {
	fast := head
	for k > 0 { // fast 先走 k 步
		fast = fast.Next
		k--
	}
	slow := head
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
	}
	return slow.Val
}

func RemoveDuplicateNodes(head *ListNode) *ListNode {
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
 * LeetCode T141 环形链表
 * 给定一个链表，判断链表中是否有环
 * https://leetcode-cn.com/problems/linked-list-cycle/
 */
// 方法 1：快慢指针法
// 假设非环部分长 N，环形部分长 K，那么时间复杂度为 O(N+K)，也就是 O(n)
func HasCycle(head *ListNode) bool {
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
func HasCycle2(head *ListNode) bool {
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
// 假设相遇时，slow 走了 K 步（起点到相遇点的距离），那两倍速的 fast 走了 2K 步
// fast 多走的部分就是环的长度，即 2K-K = K
// 假设相遇点离环开始处 M 步，那么起点距离环开始处为 K-M，因此 fast 和 slow 同步走，在交点处相遇
func DetectCycle(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
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
	slow = head // slow 从头开始走
	for slow != fast {
		slow = slow.Next
		fast = fast.Next
	}
	// 相遇到交点处
	return slow
}

/*
 * LeetCode T面试题52. 两个链表的第一个公共节点
 * https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/
 * 输入两个链表，找出它们的第一个公共节点。
 */
// 方法 1：双指针法
func GetIntersectionNode(headA, headB *ListNode) *ListNode {
	currA := headA
	currB := headB
	if currA == nil || currB == nil {
		return nil
	}
	var loop int
	for currA != currB {
		currA = currA.Next
		if currA == nil {
			currA = headB
			loop++
		}
		currB = currB.Next
		if currB == nil {
			currB = headA
			loop++
		}
		// 当两个指针同时到达结尾时，loop = 4，没有公共部分，返回 nil
		if loop > 2 {
			return nil
		}
	}
	return currA
}

// 方法 2：哈希表法

/*
 * LeetCode T876 链表的中间结点
 * https://leetcode-cn.com/problems/middle-of-the-linked-list/
 *
 * 给定一个带有头结点 head 的非空单链表，返回链表的中间结点。
 * 如果有两个中间结点，则返回第二个中间结点。
 */
// 方法 1：快慢指针法
// 时间复杂度 O(N)，空间复杂度 O(1)
func MiddleNode(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
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
func DeleteSomeNode(node *ListNode) {
	if node == nil || node.Next == nil {
		return
	}
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

// 删除链表的中间节点
// Q1：确认偶数节点的时候删哪个（前 or 后）
// Q2: 确认只有一个节点怎么处理
func DeleteMiddleNode(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	dummy := &ListNode{
		Next: head,
	}
	slow := dummy
	fast := dummy
	if fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	if slow.Next == nil {
		slow = nil
	} else {
		slow.Val = slow.Next.Val
		slow.Next = slow.Next.Next
	}
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
