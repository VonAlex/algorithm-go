package leetcode

import "testing"

func Test_MyLinkedList(t *testing.T) {
	linkedList := LinkedListConstructor()
	t.Logf("len=%d", linkedList.len)
	linkedList.AddAtTail(1)
	t.Logf("len=%d", linkedList.len)
	linkedList.AddAtTail(2)
	t.Logf("len=%d", linkedList.len)
	linkedList.AddAtTail(3)
	t.Logf("len=%d", linkedList.len)
	linkedList.AddAtTail(4)
	t.Logf("len=%d", linkedList.len)

	listPrint(linkedList.head)

	t.Log(linkedList.Get(1))
	// linkedList.AddAtIndex(0, 9)
	// listPrint(linkedList.head)

	// linkedList.DeleteAtIndex(2)
	// listPrint(linkedList.head)
}
