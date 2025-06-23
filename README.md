# deepforest-mycology

English version:

This project focuses on classifying wild mushroom species based on images, aiming to support the accurate identification of toxic mushrooms. To achieve the best classification performance, four different feature extraction and modeling strategies were implemented and evaluated.

🧪 Methods Used
1. HOG + Histogram + Random Forest
Features are extracted using Histogram of Oriented Gradients (HOG) and color histograms to capture edge direction and color distribution.

A Random Forest classifier is trained on these handcrafted features.

2. LBP (Local Binary Pattern) + Random Forest
Uses LBP to extract local texture patterns from grayscale images.

Features are fed into a Random Forest for classification.

3. Transfer Learning with EfficientNetB2 + Random Forest
Leverages a pre-trained EfficientNetB2 model to extract deep image features.

These features are passed into a Random Forest to perform classification.

4. Transfer Learning with ResNet-50 + Random Forest
Similar to the above but uses ResNet-50 as the feature extractor.

Combines the power of deep residual networks with a lightweight random forest classifier.

⚙️ Optimization & Enhancements
Hyperparameter tuning for the Random Forest classifier was performed using Optuna, an automated hyperparameter optimization framework.

Data augmentation techniques (such as rotation, flipping, and brightness adjustment) were applied to enhance the diversity of the training dataset and improve model generalization.

#Phiên bản tiếng việt:
Dự án này tập trung vào việc phân loại các loài nấm hoang dã thông qua hình ảnh, với mục tiêu hỗ trợ nhận dạng chính xác các loài nấm độc hại. Để đạt được hiệu quả phân loại tốt nhất, mình đã áp dụng bốn phương pháp xử lý đặc trưng khác nhau và so sánh hiệu suất của chúng.

🔍 Các phương pháp sử dụng
1. HOG + Histogram + Random Forest
Trích xuất đặc trưng bằng Histogram of Oriented Gradients và phân tích tần suất pixel qua histogram màu.

Áp dụng Random Forest để huấn luyện mô hình phân loại.

2. LBP (Local Binary Pattern) + Random Forest
Sử dụng LBP để trích xuất đặc trưng kết cấu cục bộ trên ảnh xám.

Kết hợp với Random Forest nhằm đánh giá hiệu quả của đặc trưng kết cấu.

3. Transfer Learning với EfficientNetB2 + Random Forest
Sử dụng mô hình EfficientNetB2 đã được huấn luyện trước trên ImageNet để trích xuất đặc trưng sâu (deep features).

Sau đó đưa vào Random Forest để phân loại.

4. Transfer Learning với ResNet-50 + Random Forest
Tương tự phương pháp trên nhưng thay EfficientNetB2 bằng ResNet-50.

ResNet nổi bật với kiến trúc residual giúp học được đặc trưng trừu tượng và ổn định.



⚙️ Tối ưu mô hình và cải tiến hiệu suất
Các tham số (hyperparameters) của mô hình Random Forest được tối ưu tự động bằng Optuna, nhằm tìm ra cấu hình tốt nhất cho từng phương pháp.

Bên cạnh đó, mình cũng áp dụng Data Augmentation (phép biến đổi ảnh như xoay, lật, thay đổi độ sáng...) để tăng độ đa dạng của dữ liệu và cải thiện khả năng tổng quát của mô hình.
