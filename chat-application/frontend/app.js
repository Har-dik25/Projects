let ws;
let currentUser = "";
let chatWith = "";
let chatType = "";

function connect() {
    currentUser = document.getElementById("username").value;
    chatWith = document.getElementById("to").value;
    chatType = document.getElementById("type").value;

    document.getElementById("login").style.display = "none";
    document.getElementById("chat-header").innerText = chatWith;

    ws = new WebSocket(`ws://localhost:8000/ws/${currentUser}`);

    ws.onmessage = (event) => {
        let data = JSON.parse(event.data);
        addMessage(data.from, data.msg);
    };
}

function sendMessage() {
    let msg = document.getElementById("msg").value;
    if (!msg) return;

    let data = {
        type: chatType,
        msg: msg
    };

    if (chatType === "private") {
        data.to = chatWith;
    } else {
        data.group = chatWith;
    }

    ws.send(JSON.stringify(data));
    addMessage(currentUser, msg);
    document.getElementById("msg").value = "";
}

function addMessage(from, msg) {
    let chat = document.getElementById("chat");
    let div = document.createElement("div");

    div.classList.add("message");
    if (from === currentUser) div.classList.add("sender");
    else div.classList.add("receiver");

    div.innerHTML = `<b>${from}</b><br>${msg}`;
    chat.appendChild(div);

    chat.scrollTop = chat.scrollHeight;
}
