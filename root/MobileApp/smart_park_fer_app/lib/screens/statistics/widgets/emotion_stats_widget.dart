import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:smart_park_fer_app/core/constants/colors.dart';
import 'package:smart_park_fer_app/models/emotion_data.dart';

class EmotionStatsWidget extends StatelessWidget {
  final String title;
  final EmotionData emotionData;

  const EmotionStatsWidget({
    super.key,
    required this.title,
    required this.emotionData,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: Colors.white,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            Text(
              title.toUpperCase(),
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.2,
                color: AppColors.charcoal,
              ),
            ),
            const SizedBox(height: 24),
            // Radar Chart Section
            SizedBox(
              height: 250,
              child: RadarChart(
                RadarChartData(
                  radarShape: RadarShape.polygon,
                  
                  dataSets: [
                    RadarDataSet(
                      fillColor: AppColors.happiness.withValues(alpha: 0.4),
                      borderColor: AppColors.happiness,
                      entryRadius: 3,
                      dataEntries: [
                        RadarEntry(value: emotionData.happiness.toDouble()),
                        RadarEntry(value: emotionData.surprise.toDouble()),
                        RadarEntry(value: emotionData.neutral.toDouble()),
                        RadarEntry(value: emotionData.sadness.toDouble()),
                        RadarEntry(value: emotionData.fear.toDouble()),
                        RadarEntry(value: emotionData.disgust.toDouble()),
                        RadarEntry(value: emotionData.contempt.toDouble()),
                        RadarEntry(value: emotionData.anger.toDouble()),
                      ],
                    ),
                  ],
                  getTitle: (index, angle) {
                    final titles = [
                      'Happiness', 'Surprise', 'Neutral', 'Sadness', 
                      'Fear', 'Disgust', 'Contempt', 'Anger'
                    ];
                    return RadarChartTitle(
                      text: titles[index],
                      angle: angle,
                    );
                  },
                  radarBorderData: const BorderSide(color: Colors.transparent),
                  tickBorderData: const BorderSide(color: Colors.transparent),
                  gridBorderData: BorderSide(color: Colors.grey.withValues(alpha: 0.2), width: 1),
                  tickCount: 5,
                  ticksTextStyle: const TextStyle(color: Colors.transparent),
                ),
              ),
            ),
            const SizedBox(height: 24),
            const Divider(color: AppColors.dividerColor, thickness: 1),
            const SizedBox(height: 24),
            // Bar Graphs Section
            _buildEmotionBar("Happiness", emotionData.happiness.toDouble(), AppColors.happiness),
            _buildEmotionBar("Surprise", emotionData.surprise.toDouble(), AppColors.surprise),
            _buildEmotionBar("Anger", emotionData.anger.toDouble(), AppColors.anger),
            _buildEmotionBar("Disgust", emotionData.disgust.toDouble(), AppColors.disgust),
            _buildEmotionBar("Fear", emotionData.fear.toDouble(), AppColors.fear),
            _buildEmotionBar("Sadness", emotionData.sadness.toDouble(), AppColors.sadness),
            _buildEmotionBar("Contempt", emotionData.contempt.toDouble(), AppColors.contempt),
            _buildEmotionBar("Neutral", emotionData.neutral.toDouble(), AppColors.neutral),
          ],
        ),
      ),
    );
  }

  Widget _buildEmotionBar(String label, double percentage, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Row(
        children: [
          Container(
            width: 12,
            height: 12,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 12),
          SizedBox(
            width: 80,
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: AppColors.charcoal,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: percentage / 100,
                backgroundColor: Colors.grey.withValues(alpha: 0.1),
                color: color,
                minHeight: 8,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            "${percentage.toInt()}%",
            style: const TextStyle(
              fontSize: 12,
              color: Colors.grey,
            ),
          ),
        ],
      ),
    );
  }
}
