document.addEventListener("DOMContentLoaded", function() {
    const sendButton = document.getElementById("send-button");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const typingIndicator = document.getElementById("typing-indicator");

    sendButton.addEventListener("click", function() {
        const message = userInput.value.trim();
        if (message) {
            appendMessage("", message);
            userInput.value = "";
            showTypingIndicator();

            fetch("/get_response", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                hideTypingIndicator();
                appendMessage("Bot", data.response);
            })
            .catch(error => {
                hideTypingIndicator();
                appendMessage("Bot", "Sorry, something went wrong.");
                console.error("Error:", error);
            });
        }
    });

    function appendMessage(sender, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message");
        if (sender === "Bot") {
            messageElement.classList.add("bot-message");
            messageElement.innerHTML = `<strong><i class="fas fa-robot bot-icon"></i>${sender}:</strong> ${message}`;
        } else {
            messageElement.classList.add("user-message");
            messageElement.innerHTML = `<strong>${sender}</strong> ${message}`;
        }
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function showTypingIndicator() {
        typingIndicator.classList.remove("hidden");
    }

    function hideTypingIndicator() {
        typingIndicator.classList.add("hidden");
    }
});
