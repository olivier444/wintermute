class UnionJoin:
    __slots__ = ("parent", "rank")
    def __init__(self):
        self.parent = {}
        self.rank = {}


    def find_root(self, x):
        p = self.parent.get(x)
        if p is None:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        
        root = p
        while True:
            new_root = self.parent[root]
            if new_root == root:
                break
            root = new_root

        while True:
            next_x = self.parent[x]
            self.parent[x] = root
            if next_x == x:
                break

            x = next_x

        return root


    def link(self, a, b):
        ra, rb = self.find_root(a), self.find_root(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        return