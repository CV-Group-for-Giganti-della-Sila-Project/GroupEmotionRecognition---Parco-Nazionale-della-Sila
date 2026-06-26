enum MessageSender { user, silvan }

class ChatMessage {
  final String content;
  final MessageSender sender;
  final String? imagePath; // Optional path for image messages
  final DateTime timestamp;

  ChatMessage({
    required this.content,
    required this.sender,
    this.imagePath,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}
