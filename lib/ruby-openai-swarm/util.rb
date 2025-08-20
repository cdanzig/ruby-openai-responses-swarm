module OpenAISwarm
  module Util
    class << self
      def latest_role_user_message(history)
        return history if history.empty?
        filtered_messages = symbolize_keys_to_string(history.dup)
        last_user_message = filtered_messages.reverse.find { |msg| msg['role'] == 'user' }
        last_user_message ? [last_user_message] : history
      end

      def request_tools_excluded(tools, tool_names, prevent_agent_reentry = false)
        return nil if tools.empty?
        return tools if tool_names.empty? || !prevent_agent_reentry

        symbolize_keys_to_string(tools).reject do |tool|
          tool_names.include?("#{tool['name']}")
        end
      end
    end

    def self.debug_print(debug, *args)
      return unless debug
      timestamp = Time.now.strftime("%Y-%m-%d %H:%M:%S")
      message = args.map(&:to_s).join(' ')
      puts "\e[97m[\e[90m#{timestamp}\e[97m]\e[90m \n\n#{message}\n \e[0m"
    end

    def self.symbolize_keys_to_string(obj)
      case obj
      when Hash
        obj.transform_keys(&:to_s).transform_values { |v| symbolize_keys_to_string(v) }
      when Array
        obj.map { |v| symbolize_keys_to_string(v) }
      else
        obj
      end
    end

    def self.clean_message_tools(messages, tool_names)
      return messages if tool_names.empty?
      filtered_messages = symbolize_keys_to_string(messages.dup)
      call_ids_to_remove = filtered_messages.select { |msg| msg['type'] == 'function_call' and tool_names.include?(msg['name']) }.map { |msg| msg['call_id'] }

      # Remove specific messages
      filtered_messages.each_with_index do |msg, index|
        # Remove function_call messages for specified function names
        if msg['type'] == 'function_call' && tool_names.include?(msg['name'])
          Util.debug_print(true, "Cleaning #{msg['name']} function_call (call_id #{msg['call_id']}) \n #{msg.inspect}")
          filtered_messages[index] = nil

          if filtered_messages[index-1] && filtered_messages[index-1]["type"] == 'reasoning'
            Util.debug_print(true, "Cleaning associated reasoning message \n #{filtered_messages[index-1].inspect}")
            filtered_messages[index-1] = nil # remove the reasoning message
          end
        end

        # Remove function_call_output responses for associated function_call messages
        if msg['type'] == 'function_call_output' && call_ids_to_remove.include?(msg['call_id'])
          Util.debug_print(true, "Cleaning #{msg['type']} for call_id #{msg['call_id']} \n #{msg.inspect}")
          filtered_messages[index] = nil
        end
      end

      # filtered_messages.compact
    end

    def self.message_template(agent_name)
      {
        "content" => "",
        "sender" => agent_name,
        "role" => "assistant",
        "function_call" => nil,
        "tool_calls" => Hash.new do |hash, key|
          hash[key] = {
            "function" => { "arguments" => "", "name" => "" },
            "id" => "",
            "type" => ""
          }
        end
      }
    end

    def self.merge_fields(target, source)
      semantic_keyword = %W[type]
      source.each do |key, value|
        if value.is_a?(String)
          if semantic_keyword.include?(key)
            target[key] = value
          else
            target[key] += value
          end
        elsif value.is_a?(Hash) && value != nil
          merge_fields(target[key], value)
        end
      end
    end

    def self.merge_chunk(final_response, delta)
      delta.delete("role")
      merge_fields(final_response, delta)

      tool_calls = delta["tool_calls"]
      if tool_calls && !tool_calls.empty?
        index = tool_calls[0].delete("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])
      end
    end

    def self.function_to_json(func_instance)
      # is this a built in OpenAI Tool?
      if func_instance.is_a?(Hash) && (func_instance[:type] == "web_search_preview" || func_instance[:type] == "file_search")
        Util.debug_print(true, "Found a built in OpenAI Tool: #{func_instance.inspect}")
        return func_instance
      end

      is_target_method = func_instance.respond_to?(:target_method) || func_instance.is_a?(OpenAISwarm::FunctionDescriptor)
      func = is_target_method ? func_instance.target_method : func_instance
      custom_parameters = is_target_method ? func_instance.parameters : nil

      function_name = func.name
      function_parameters = func.parameters

      type_map = {
        String => "string",
        Integer => "integer",
        Float => "number",
        TrueClass => "boolean",
        FalseClass => "boolean",
        Array => "array",
        Hash => "object",
        NilClass => "null"
      }
      parameters = {}

      function_parameters.each do |type, param_name|
        param_type = type_map[param_name.class] || "string"
        if param_name.to_s == 'context_variables' && type == :opt #type == :keyreq
          param_type = 'object'
        end
        parameters[param_name] = { type: param_type }
      end

      required = function_parameters
        .select { |type, _| [:req, :keyreq].include?(type) }
        .map { |_, name| name.to_s }

      description = func_instance.respond_to?(:description) ? func_instance&.description : nil

      json_parameters = {
        type: "object",
        properties: parameters,
        required: required
      }

      {
        type: "function",
        name: function_name,
        description: description || '',
        parameters: custom_parameters || json_parameters
      }
    end
  end
end
