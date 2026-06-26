import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:smart_park_fer_app/core/constants/colors.dart';
import 'package:smart_park_fer_app/core/services/chat_service.dart';
import 'package:smart_park_fer_app/models/chat_message.dart';
import 'package:smart_park_fer_app/screens/chat/widgets/chat_bubbles.dart';
import 'package:smart_park_fer_app/screens/chat/widgets/chat_input_bar.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final ScrollController _scrollController = ScrollController();
  final List<ChatMessage> _messages = [
    ChatMessage(
      content: "Hello, i'm Silvan. Send me a photo: i'll guess the group emotion. You can also ask me about the emotions felt in the park!",
      sender: MessageSender.silvan,
    ),
  ];

  bool _isTyping = false;

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _sendMessage(String text, String? imagePath) async {
    final userMessage = ChatMessage(
      content: text,
      sender: MessageSender.user,
      imagePath: imagePath,
    );

    setState(() {
      _messages.add(userMessage);
      _isTyping = true;
    });
    _scrollToBottom();

    try {
      String responseContent;
      if (imagePath != null) {
        // Convert image to base64 and call analyzePhoto
        final File imageFile = File(imagePath);
        final List<int> imageBytes = await imageFile.readAsBytes();
        final String base64Image = base64Encode(imageBytes);
        final String emotion = await ChatService().analyzePhoto(base64Image);
        responseContent = "Detected emotion: $emotion";
      } else {
        // Call askAgent for text messages
        responseContent = await ChatService().askAgent(text);
      }

      if (mounted) {
        setState(() {
          _isTyping = false;
          _messages.add(ChatMessage(
            content: responseContent,
            sender: MessageSender.silvan,
          ));
        });
        _scrollToBottom();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isTyping = false;
          _messages.add(ChatMessage(
            content: "Sorry, I encountered an error: $e",
            sender: MessageSender.silvan,
          ));
        });
        _scrollToBottom();
      }
    }
  }


  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.creamBg,
      body: Stack(
        children: [
          Positioned(
            top: -20,
            right: -20,
            child: Icon(
              Icons.eco,
              size: 150,
              color: AppColors.forestGreen.withValues(alpha: 0.05),
            ),
          ),
          Positioned(
            bottom: 100,
            left: -30,
            child: Icon(
              Icons.eco,
              size: 200,
              color: AppColors.forestGreen.withValues(alpha: 0.05),
            ),
          ),
          SafeArea(
            child: Column(
              children: [
                const SizedBox(height: 20),
                Expanded(
                  child: ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemCount: _messages.length + (_isTyping ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (index == _messages.length) {
                        return const Padding(
                          padding: EdgeInsets.only(bottom: 24.0),
                          child: TypingBubble(),
                        );
                      }

                      final msg = _messages[index];
                      
                      if (msg.sender == MessageSender.user) {
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 24.0),
                          child: UserBubble(message: msg),
                        );
                      } else {
                        // Check if this is the start of a Silvan group
                        // (either it's the first message or the previous one wasn't from Silvan)
                        bool isStartOfGroup = index == 0 || _messages[index - 1].sender != MessageSender.silvan;
                        
                        if (!isStartOfGroup) {
                          return const SizedBox.shrink(); // Handled by the group starter
                        }

                        // Group consecutive Silvan messages
                        List<ChatMessage> silvanGroup = [msg];
                        int nextIndex = index + 1;
                        while (nextIndex < _messages.length && 
                               _messages[nextIndex].sender == MessageSender.silvan) {
                          silvanGroup.add(_messages[nextIndex]);
                          nextIndex++;
                        }
                        
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 24.0),
                          child: AgentBubble(messages: silvanGroup),
                        );
                      }
                    },
                  ),
                ),
                ChatInputBar(onSendMessage: _sendMessage),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
