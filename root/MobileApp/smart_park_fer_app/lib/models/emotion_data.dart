class EmotionData {
  final int happiness;
  final int surprise;
  final int neutral;
  final int sadness;
  final int fear;
  final int disgust;
  final int contempt;
  final int anger;

  EmotionData({
    this.happiness = 0,
    this.surprise = 0,
    this.neutral = 0,
    this.sadness = 0,
    this.fear = 0,
    this.disgust = 0,
    this.contempt = 0,
    this.anger = 0,
  });

  factory EmotionData.fromJson(Map<String, dynamic> json) {
    return EmotionData(
      happiness: json['happiness'] ?? 0,
      surprise: json['surprise'] ?? 0,
      neutral: json['neutral'] ?? 0,
      sadness: json['sadness'] ?? 0,
      fear: json['fear'] ?? 0,
      disgust: json['disgust'] ?? 0,
      contempt: json['contempt'] ?? 0,
      anger: json['anger'] ?? 0,
    );
  }
}
