class TicketMS:
    def __init__(self, capacity=100):
        self.data = [None] * capacity
        self.front = -1
        self.rear = -1
        self.size = 0
        self.capacity = capacity
        self.ticket_id = 1

    def is_emptyqueue(self):
        return self.size == 0

    def is_fullqueue(self):
        return self.size == self.capacity

    def addticket(self, issue):
        if self.is_fullqueue():
            print("Queue is full")
            return

        ticket = {
            "ID": self.ticket_id,
            "Issue": issue
        }
        self.ticket_id += 1

        if self.is_emptyqueue():
            self.front = 0
            self.rear = 0
        else:
            self.rear += 1

        self.data[self.rear] = ticket
        self.size += 1
        print(f"Ticket added: {ticket}")

    def deleteticket(self):
        if self.is_emptyqueue():
            print("No tickets to resolve")
            return None

        ticket = self.data[self.front]

        if self.front == self.rear:
            self.front = -1
            self.rear = -1
        else:
            self.front += 1

        self.size -= 1
        return ticket

    def peekfromqueue(self):
        if self.is_emptyqueue():
            print("Queue is empty")
            return None
        return self.data[self.front]

    def traversetickets(self):
        if self.is_emptyqueue():
            print("No pending tickets")
            return

        print("\nPending Tickets:")
        for i in range(self.front, self.rear + 1):
            print(self.data[i])

    def get_numofticks(self):
        return self.size


def run_cli():
    system = TicketMS(10)

    print("\nTicket Support System (FIFO Queue)")
    print("------------------------------------")
    print("Commands:")
    print(" add <issue>   -> Raise new ticket")
    print(" delete        -> Resolve ticket")
    print(" peek          -> View next ticket")
    print(" list          -> Show all tickets")
    print(" size          -> Total tickets")
    print(" exit          -> Quit program")

    while True:
        user_input = input("\n> ").strip().split(maxsplit=1)
        if not user_input:
            continue

        cmd = user_input[0].lower()
        arg = user_input[1] if len(user_input) > 1 else None

        if cmd == "add":
            if arg:
                system.addticket(arg)
            else:
                print("Please provide issue description")

        elif cmd == "delete":
            ticket = system.deleteticket()
            if ticket:
                print(f"Ticket resolved: {ticket}")

        elif cmd == "peek":
            ticket = system.peekfromqueue()
            if ticket:
                print(f"⏭Next ticket: {ticket}")

        elif cmd == "list":
            system.traversetickets()

        elif cmd == "size":
            print(f"Total tickets: {system.get_numofticks()}")

        elif cmd == "exit":
            print("Exiting Ticket System")
            break

        else:
            print("Invalid command. Try again.")


if __name__ == "__main__":
    run_cli()
