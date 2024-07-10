document.addEventListener('DOMContentLoaded', function () {
    var chatContainer = document.getElementById('chat-container');
    var chatBox = document.getElementById('chat-box');
    var messageInput = document.getElementById('message');
    var sendButton = document.getElementById('send-btn');

    sendButton.addEventListener('click', function () {
        sendMessage();
    });

    messageInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        var message = messageInput.value;
        if (message.trim() === '') return;

        displayMessage('You: ' + message);
        messageInput.value = '';

        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        })
            .then(response => response.json())
            .then(data => {
                displayMessage('Chatbot: ' + data.response);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    }

    function displayMessage(message) {
        var messageElement = document.createElement('div');
        messageElement.innerText = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
