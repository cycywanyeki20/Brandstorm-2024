# Fine-tune the model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with new hyperparameters
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=5,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32
)
