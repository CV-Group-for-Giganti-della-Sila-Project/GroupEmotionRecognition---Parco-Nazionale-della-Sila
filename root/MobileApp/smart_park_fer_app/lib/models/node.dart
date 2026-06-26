import 'package:smart_park_fer_app/models/emotion_data.dart';

class Node {
  final String id;
  final String name;
  final EmotionData emotionData;

  Node({
    required this.id,
    required this.name,
    required this.emotionData,
  });
}
