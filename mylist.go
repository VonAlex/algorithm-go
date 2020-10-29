package leetcode

// 标记长度，避免不必要的遍历
type MyLinkedList struct {
	head *ListNode
	len  int
}

/** Initialize your data structure here. */
func LinkedListConstructor() MyLinkedList {
	return MyLinkedList{}
}

// index 从 0 开始
/** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
func (this *MyLinkedList) Get(index int) int {
	if index < 0 || index >= this.len {
		return -1
	}
	curr := this.head
	for i := 0; i < index; i++ {
		curr = curr.Next
	}
	return curr.Val
}

/** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
func (this *MyLinkedList) AddAtHead(val int) {
	node := &ListNode{
		Val:  val,
		Next: this.head,
	}
	this.head = node
	this.len++
	return
}

/** Append a node of value val to the last element of the linked list. */
func (this *MyLinkedList) AddAtTail(val int) {
	node := &ListNode{
		Val: val,
	}
	curr := this.head
	if curr == nil {
		this.head = node
		this.len++
		return
	}
	for curr.Next != nil {
		curr = curr.Next
	}
	curr.Next = node
	this.len++
	return
}

/** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
// 如果 index 等于链表的长度，则该节点将附加到链表的末尾。
// 如果 index 大于链表长度，则不会插入节点。
// 如果index 小于 0，则在头部插入节点。
func (this *MyLinkedList) AddAtIndex(index int, val int) {
	if index <= 0 {
		this.AddAtHead(val)
		return
	}
	if index > this.len {
		return
	}
	if index == this.len {
		this.AddAtTail(val)
		return
	}
	// 插入和删除时使用 prev 记录要操作结点的前一个结点
	var prev *ListNode
	curr := this.head
	for i := 0; i < index; i++ {
		prev = curr
		curr = curr.Next
	}
	node := &ListNode{
		Val:  val,
		Next: curr,
	}
	prev.Next = node
	this.len++
	return
}

/** Delete the index-th node in the linked list, if the index is valid. */
func (this *MyLinkedList) DeleteAtIndex(index int) {
	if index < 0 || index >= this.len {
		return
	}
	curr := this.head
	if index == 0 {
		this.head = curr.Next
		this.len--
		return
	}

	var prev *ListNode
	for i := 0; i < index; i++ {
		prev = curr
		curr = curr.Next
	}
	prev.Next = curr.Next
	this.len--
	return
}

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * obj := Constructor();
 * param_1 := obj.Get(index);
 * obj.AddAtHead(val);
 * obj.AddAtTail(val);
 * obj.AddAtIndex(index,val);
 * obj.DeleteAtIndex(index);
 */
